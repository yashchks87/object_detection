#!/usr/bin/env python
"""Fast, restart-safe upload of a local directory tree to a Unity Catalog Volume.

Uses the Databricks Files API (HTTPS PUT /api/2.0/fs/files/Volumes/...) instead
of the /Volumes FUSE mount. FUSE serialises file creation (~5 files/s for small
files regardless of thread count); the Files API handles concurrent uploads
(~370 files/s measured with 128 threads for ~160 KB COCO JPEGs, i.e. the full
118k-image train2017 in ~5-6 min). Throughput is capped server-side, so going
far beyond ~128 threads does not help.

How it works:
    1. Scan --src locally; list --dst via the Files API (name + size).
    2. Upload every file that is missing on --dst or has a different size,
       using a thread pool with pooled keep-alive HTTPS connections.
       Transient failures (HTTP 429/5xx, connection errors) are retried with
       exponential backoff + jitter, honouring Retry-After.
    3. Re-list --dst and verify every file's size. Mismatches are re-uploaded
       in another pass (up to --max-passes).

Guarantees:
    - Restart-safe: files already on --dst with the same size are skipped, so
      an interrupted run can simply be re-run.
    - Verified: success is only reported after the server-side listing matches
      the local tree (names + sizes). Exit code is non-zero otherwise, and the
      failing paths are written to --failed-list.
    - Live tqdm progress bar (files, files/s, ETA, GB, MB/s, counts).

Auth: standard Databricks unified auth via databricks-sdk (DATABRICKS_HOST +
DATABRICKS_TOKEN env vars on a cluster, or a ~/.databrickscfg profile via
DATABRICKS_CONFIG_PROFILE). Files must be <= 5 GiB (single-PUT API limit).

Usage:
    python copy_to_volume.py \\
        --src /local_disk0/train2017 \\
        --dst /Volumes/daai_ke_team/default/images/object_detection_datasets/coco/train

Background run (survives ssh disconnect):
    nohup python -u copy_to_volume.py --src ... --dst ... > copy.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from databricks.sdk.core import Config
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger("copy_to_volume")

MAX_PUT_BYTES = 5 * 1024**3
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class NonRetryableError(Exception):
    pass


class FilesAPI:
    """Minimal thread-safe client for the Databricks Files API."""

    def __init__(self, pool_size: int, timeout: float, retries: int):
        self.cfg = Config()
        self.base = f"{self.cfg.host.rstrip('/')}/api/2.0/fs"
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        # One pooled keep-alive connection per worker thread. No urllib3-level
        # retries: a retried streamed body would be empty; we retry by reopening.
        self.session.mount("https://", HTTPAdapter(pool_connections=1, pool_maxsize=pool_size, max_retries=0))

    def _request(self, method: str, url: str, body_path: Path | None = None,
                 headers: dict | None = None, **kwargs) -> requests.Response:
        for attempt in range(1, self.retries + 1):
            try:
                req_headers = {**self.cfg.authenticate(), **(headers or {})}
                if body_path is None:
                    resp = self.session.request(method, url, headers=req_headers, timeout=self.timeout, **kwargs)
                else:
                    with open(body_path, "rb") as f:
                        resp = self.session.request(method, url, headers=req_headers, data=f,
                                                    timeout=self.timeout, **kwargs)
                if resp.status_code < 400 or resp.status_code == 404:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS:
                    raise NonRetryableError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                err = f"HTTP {resp.status_code}"
                delay = float(resp.headers.get("Retry-After", 0) or 0)
            except requests.RequestException as exc:
                err, delay = f"{type(exc).__name__}: {exc}", 0.0
            if attempt == self.retries:
                raise RuntimeError(f"{method} failed after {attempt} attempts: {err}")
            time.sleep(max(delay, min(60.0, 2 ** attempt)) * random.uniform(0.8, 1.2))
        raise AssertionError("unreachable")

    def list_tree(self, root: str) -> dict[str, int]:
        """Recursively list root -> {relative_path: size}. Missing root -> {}."""
        out: dict[str, int] = {}
        stack = [root.rstrip("/")]
        while stack:
            directory = stack.pop()
            token = None
            while True:
                params = {"page_size": 1000, **({"page_token": token} if token else {})}
                resp = self._request("GET", f"{self.base}/directories{quote(directory)}", params=params)
                if resp.status_code == 404:
                    break
                body = resp.json()
                for entry in body.get("contents", []):
                    if entry["is_directory"]:
                        stack.append(entry["path"].rstrip("/"))
                    else:
                        out[os.path.relpath(entry["path"], root)] = entry["file_size"]
                token = body.get("next_page_token")
                if not token:
                    break
        return out

    def upload(self, local: Path, remote: str) -> None:
        # Parent directories are created implicitly by the Files API.
        resp = self._request("PUT", f"{self.base}/files{quote(remote)}", body_path=local,
                             params={"overwrite": "true"},
                             headers={"Content-Type": "application/octet-stream"})
        if resp.status_code == 404:
            raise NonRetryableError(f"HTTP 404 for {remote} (volume missing or no WRITE VOLUME permission?)")


def scan_tree(root: Path) -> dict[str, int]:
    """Return {relative_path: size} for every regular file under root."""
    files: dict[str, int] = {}
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files[os.path.relpath(entry.path, root)] = entry.stat().st_size
    return files


def upload_all(api: FilesAPI, src: Path, dst: str, todo: list[tuple[str, int]], workers: int,
               log_every: float, pass_idx: int) -> list[str]:
    """Upload todo files concurrently; returns relative paths that failed."""
    failed: list[str] = []
    counts = {"ok": 0, "failed": 0}
    total_bytes, done_bytes = sum(s for _, s in todo), 0
    t_start = time.time()
    interactive = sys.stderr.isatty()

    with (
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="upload") as pool,
        logging_redirect_tqdm(),
        tqdm(total=len(todo), unit="file", desc=f"upload pass {pass_idx}", dynamic_ncols=True,
             smoothing=0.05, mininterval=0.5 if interactive else log_every) as bar,
    ):
        futures = {pool.submit(api.upload, src / rel, f"{dst}/{rel}"): (rel, size) for rel, size in todo}
        try:
            for fut in as_completed(futures):
                rel, size = futures[fut]
                try:
                    fut.result()
                    counts["ok"] += 1
                except Exception as exc:
                    LOG.error("FAILED %s: %s", rel, exc)
                    counts["failed"] += 1
                    failed.append(rel)
                done_bytes += size
                elapsed = time.time() - t_start
                bar.set_postfix(
                    ok=counts["ok"], failed=counts["failed"],
                    GB=f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f}",
                    MBps=f"{done_bytes / elapsed / 1e6:.1f}" if elapsed else "0.0",
                    refresh=False,
                )
                bar.update(1)
        except KeyboardInterrupt:
            LOG.warning("Interrupted: cancelling pending uploads (in-flight ones finish first)...")
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    LOG.info("Pass %d: uploaded %d files (%.2f GB) in %.1f min, %d failed",
             pass_idx, counts["ok"], total_bytes / 1e9, (time.time() - t_start) / 60, counts["failed"])
    return failed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("/local_disk0/train2017"))
    p.add_argument("--dst", default="/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/train",
                   help="Destination directory inside a UC Volume (must start with /Volumes/).")
    p.add_argument("--workers", type=int, default=128,
                   help="Concurrent uploads. ~128 is the measured sweet spot; more is slower.")
    p.add_argument("--retries", type=int, default=6, help="Attempts per HTTP request.")
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (seconds).")
    p.add_argument("--max-passes", type=int, default=3,
                   help="Upload+verify passes; later passes only re-upload missing/mismatched files.")
    p.add_argument("--limit", type=int, default=None, help="Upload only the first N files (smoke testing).")
    p.add_argument("--log-every", type=float, default=30.0,
                   help="Progress bar refresh interval (seconds) when stderr is not a TTY (e.g. nohup).")
    p.add_argument("--failed-list", type=Path, default=Path("copy_failed.txt"),
                   help="Where to write relative paths that are still missing/mismatched at the end.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    args = parse_args()
    dst = args.dst.rstrip("/")
    LOG.info("Args: %s", vars(args))
    if not args.src.is_dir():
        LOG.error("Source directory does not exist: %s", args.src)
        return 2
    if not dst.startswith("/Volumes/"):
        LOG.error("--dst must be a Unity Catalog Volume path starting with /Volumes/: %s", dst)
        return 2

    t0 = time.time()
    local = scan_tree(args.src)
    if args.limit:
        local = dict(sorted(local.items())[: args.limit])
    too_big = [rel for rel, size in local.items() if size > MAX_PUT_BYTES]
    if too_big:
        LOG.error("%d file(s) exceed the 5 GiB single-PUT limit, e.g. %s", len(too_big), too_big[:3])
        return 2
    LOG.info("Local: %d files (%.2f GB) under %s", len(local), sum(local.values()) / 1e9, args.src)

    api = FilesAPI(pool_size=args.workers, timeout=args.timeout, retries=args.retries)
    LOG.info("Workspace: %s (auth: %s)", api.cfg.host, api.cfg.auth_type)

    mismatched: list[str] = []
    for pass_idx in range(1, args.max_passes + 1):
        t_list = time.time()
        remote = api.list_tree(dst)
        mismatched = sorted(rel for rel, size in local.items() if remote.get(rel) != size)
        LOG.info("Remote listing: %d files in %.1fs -> %d already present, %d to upload",
                 len(remote), time.time() - t_list, len(local) - len(mismatched), len(mismatched))
        if not mismatched:
            break
        upload_all(api, args.src, dst, [(rel, local[rel]) for rel in mismatched],
                   args.workers, args.log_every, pass_idx)
    else:
        remote = api.list_tree(dst)
        mismatched = sorted(rel for rel, size in local.items() if remote.get(rel) != size)

    if mismatched:
        args.failed_list.write_text("\n".join(mismatched) + "\n")
        LOG.error("%d file(s) still missing or size-mismatched on %s after %d passes; list written to %s. "
                  "Re-run to retry (verified files are skipped).",
                  len(mismatched), dst, args.max_passes, args.failed_list.resolve())
        return 1
    LOG.info("DONE in %.1f min: all %d files verified on %s (names + sizes match).",
             (time.time() - t0) / 60, len(local), dst)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.warning("Aborted by user. Re-run the same command to resume; verified files are skipped.")
        sys.exit(130)
