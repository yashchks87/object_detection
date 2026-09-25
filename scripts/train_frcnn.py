"""Faster R-CNN training on the COCO 2017 MDS shards (torchrun DDP + W&B).

Streams samples from the MDS shards on the UC Volume through a node-local
cache, trains with DDP via torchrun, evaluates the official COCO bbox metrics
(pycocotools) on val2017 every --val-every epochs, and logs to Weights &
Biases from the primary rank.

Single GPU:
    python scripts/train_frcnn.py --cache /local_disk0/mds_cache_coco \
        --out /local_disk0/runs/frcnn_001

Multi-GPU (one process per GPU; normally via scripts/run_training.sh):
    torchrun --standalone --nproc_per_node=4 scripts/train_frcnn.py \
        --cache /local_disk0/mds_cache_coco --out /local_disk0/runs/frcnn_001

Outputs under --out: config.json, metrics.jsonl, best.pt (highest val AP;
EMA weights when --ema), last.pt (full resume state, refreshed every epoch
and every --checkpoint-every optimizer steps, with the exact streaming
position so a mid-epoch resume continues the same shuffle), optional
epoch_*.pt, and _TRAINING_SUCCESS. `--resume auto` picks up <out>/last.pt
and continues the same W&B run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.coco_data import COCODetectionDataset, collate_detection, load_dataset_meta, validate_paths
from scripts.coco_metrics import STAT_NAMES, evaluate_coco, load_coco_gt
from scripts.coco_models import (
    MODELS,
    ModelEMA,
    build_model,
    parameter_groups,
    prefetch_backbone_weights,
)

DEFAULT_MDS = Path('/Volumes/daai_ke_team/default/images/object_detection_datasets/coco/mds_shards')
DEFAULT_VAL_ANNOTATIONS = DEFAULT_MDS.parent / 'annotations' / 'instances_val2017.json'
REFERENCE_BATCH = 16  # global batch the --lr defaults are quoted at (linear scaling rule)
LOSS_KEYS = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
OPTIMIZER_DEFAULTS = {'sgd': {'lr': 0.02, 'weight_decay': 1e-4},
                      'adamw': {'lr': 1e-4, 'weight_decay': 0.05}}
MAX_CONSECUTIVE_SKIPS = 50


# --------------------------------------------------------------------------- #
# Distributed plumbing
# --------------------------------------------------------------------------- #
class Cluster:
    def __init__(self, rank=0, world_size=1, local_rank=0, local_world_size=1):
        self.rank, self.world_size = rank, world_size
        self.local_rank, self.local_world_size = local_rank, local_world_size

    @property
    def distributed(self):
        return self.world_size > 1

    @property
    def primary(self):
        return self.rank == 0

    @property
    def num_nodes(self):
        return max(1, self.world_size // self.local_world_size)

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def sum_(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.distributed:
            dist.all_reduce(tensor)
        return tensor

    def gather(self, rows):
        if not self.distributed:
            return list(rows)
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, list(rows))
        return [row for part in gathered for row in part]


def cluster_from_environment() -> Cluster:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', str(world_size)))
    if world_size < 1 or not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
        raise ValueError('Inconsistent WORLD_SIZE/RANK/LOCAL_RANK; launch one process or use torchrun.')
    if world_size > 1 and not {'MASTER_ADDR', 'MASTER_PORT'} <= set(os.environ):
        raise ValueError('Distributed training needs MASTER_ADDR/MASTER_PORT; launch with torchrun.')
    return Cluster(rank, world_size, local_rank, local_world_size)


def resolve_device(cluster: Cluster, device_arg: str) -> torch.device:
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if device_arg == 'auto' else device_arg)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; pass --device cpu or use a GPU node.')
        index = cluster.local_rank if cluster.distributed else (device.index or 0)
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {index} unavailable on this host.')
        device = torch.device('cuda', index)
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise ValueError('--device must select cpu or cuda.')
    return device


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data = parser.add_argument_group('data')
    data.add_argument('--mds', type=Path, default=DEFAULT_MDS,
                      help='Root of the MDS shards (contains train/, val/, dataset_meta.json).')
    data.add_argument('--val-annotations', type=Path, default=DEFAULT_VAL_ANNOTATIONS,
                      help='COCO instances json used as COCOeval ground truth.')
    data.add_argument('--cache', type=Path, required=True,
                      help='Node-local shard cache; never on /Volumes. Shared by all ranks.')
    data.add_argument('--out', type=Path, required=True,
                      help='New run directory under an existing parent (or the run to --resume).')
    data.add_argument('--num-workers', type=int, default=8,
                      help='Train loader workers per rank (val uses half). JPEG decode is cheap; '
                           'keep ranks x workers below the core count.')
    data.add_argument('--cache-limit', default=None, help="Optional streaming cache budget, e.g. '100gb'.")
    data.add_argument('--hflip-prob', type=float, default=0.5)

    model = parser.add_argument_group('model')
    model.add_argument('--model', choices=MODELS, default='fasterrcnn_resnet50_fpn_v2')
    model.add_argument('--pretrained-backbone', action=argparse.BooleanOptionalAction, default=True,
                       help='ImageNet-pretrained backbone (never COCO detection weights).')
    model.add_argument('--trainable-backbone-layers', type=int, default=3, choices=range(6),
                       help='Backbone stages to train, counted from the top (0-5).')
    model.add_argument('--min-size', type=int, nargs='+', default=[640, 672, 704, 736, 768, 800],
                       help='Train-time shorter-side sizes (random choice per batch = multi-scale). '
                            'The LARGEST value is used for evaluation.')
    model.add_argument('--max-size', type=int, default=1333, help='Longer-side cap.')
    model.add_argument('--sync-bn', action=argparse.BooleanOptionalAction, default=None,
                       help='Convert BatchNorm to SyncBatchNorm under DDP. Default: on for the v2 '
                            'model (trainable BN everywhere), off otherwise.')

    optim = parser.add_argument_group('optimisation')
    optim.add_argument('--epochs', type=int, default=26)
    optim.add_argument('--batch-size', type=int, default=4,
                       help='Per-GPU batch size. Measured on A10G (23GB), v2 model, bf16: bs=4 peaks '
                            'at ~7GB (worst-case padding), ~11 img/s/GPU; throughput is flat from bs=2 '
                            'to 8, so 4 x 4 GPUs = the reference global batch of 16.')
    optim.add_argument('--grad-accum', type=int, default=1)
    optim.add_argument('--optimizer', choices=tuple(OPTIMIZER_DEFAULTS), default='sgd')
    optim.add_argument('--lr', type=float, default=None,
                       help=f'Peak LR at a global batch of {REFERENCE_BATCH}, scaled linearly with the '
                            'real global batch. Default: 0.02 (sgd) / 1e-4 (adamw).')
    optim.add_argument('--momentum', type=float, default=0.9, help='SGD only.')
    optim.add_argument('--weight-decay', type=float, default=None,
                       help='Default: 1e-4 (sgd) / 0.05 (adamw).')
    optim.add_argument('--norm-weight-decay', type=float, default=0.0,
                       help='Weight decay for normalisation-layer parameters.')
    optim.add_argument('--lr-schedule', choices=('multistep', 'cosine'), default='multistep')
    optim.add_argument('--lr-steps', type=int, nargs='+', default=[16, 22],
                       help='multistep: epochs after which the LR is multiplied by --lr-gamma.')
    optim.add_argument('--lr-gamma', type=float, default=0.1)
    optim.add_argument('--min-lr-ratio', type=float, default=0.0, help='cosine: final LR / peak LR.')
    optim.add_argument('--warmup-iters', type=int, default=1000)
    optim.add_argument('--warmup-factor', type=float, default=1e-3)
    optim.add_argument('--clip-grad', type=float, default=0.0,
                       help='Max global grad norm; 0 disables clipping (the norm is still logged).')
    optim.add_argument('--ema', action=argparse.BooleanOptionalAction, default=False,
                       help='Keep an EMA of the weights; validation and best.pt then use the EMA.')
    optim.add_argument('--ema-decay', type=float, default=0.9998)
    optim.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                       help='Mixed precision (bf16 when supported, else fp16 with loss scaling).')
    optim.add_argument('--seed', type=int, default=42)
    optim.add_argument('--device', default='auto')

    evaluation = parser.add_argument_group('evaluation')
    evaluation.add_argument('--val-every', type=int, default=1, help='Validate every N epochs.')
    evaluation.add_argument('--detections-per-image', type=int, default=100)
    evaluation.add_argument('--score-threshold', type=float, default=0.05,
                            help='Minimum detection score kept at inference (COCO standard 0.05).')
    evaluation.add_argument('--nms-threshold', type=float, default=0.5)

    run = parser.add_argument_group('checkpointing / run control')
    run.add_argument('--resume', default=None,
                     help="last.pt to continue from, or 'auto' = <out>/last.pt if it exists.")
    run.add_argument('--checkpoint-every', type=int, default=2000,
                     help='Also refresh last.pt every N optimizer steps mid-epoch (0 = epoch end only).')
    run.add_argument('--save-epochs', action=argparse.BooleanOptionalAction, default=False,
                     help='Keep a weights checkpoint per epoch besides best.pt/last.pt.')
    run.add_argument('--max-train-batches', type=int, help='Debug-only training batch cap per rank/epoch.')
    run.add_argument('--val-max-batches', type=int, help='Debug-only validation batch cap per rank.')
    run.add_argument('--log-every', type=int, default=20, help='Step-level logging interval (optimizer steps).')
    run.add_argument('--no-progress', action='store_true',
                     help='Disable tqdm; periodic JSON step lines are printed instead.')

    wandb_group = parser.add_argument_group('weights & biases')
    wandb_group.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                             help='Log to Weights & Biases (primary rank only).')
    wandb_group.add_argument('--wandb-project', default='COCO object detection')
    wandb_group.add_argument('--wandb-entity', default='yashchks87')
    wandb_group.add_argument('--wandb-run-name', default=None, help='Defaults to the output directory name.')
    wandb_group.add_argument('--wandb-group', default=None)
    wandb_group.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    wandb_group.add_argument('--wandb-tags', nargs='*', default=[], metavar='TAG')
    wandb_group.add_argument('--wandb-num-images', type=int, default=16,
                             help='Validation images logged with predicted + GT boxes per evaluation.')
    wandb_group.add_argument('--wandb-image-score', type=float, default=0.4,
                             help='Minimum score for boxes drawn on the logged images.')
    wandb_group.add_argument('--wandb-log-model', action='store_true',
                             help='Upload best.pt as a W&B model artifact at the end of the run.')
    return parser


def validate_args(args) -> Cluster:
    defaults = OPTIMIZER_DEFAULTS[args.optimizer]
    args.lr = defaults['lr'] if args.lr is None else args.lr
    args.weight_decay = defaults['weight_decay'] if args.weight_decay is None else args.weight_decay
    if args.sync_bn is None:
        args.sync_bn = args.model == 'fasterrcnn_resnet50_fpn_v2'
    args.min_size = sorted(set(args.min_size))
    for name in ('epochs', 'batch_size', 'grad_accum', 'val_every', 'log_every', 'max_size',
                 'detections_per_image'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    for name in ('warmup_iters', 'checkpoint_every', 'num_workers', 'wandb_num_images'):
        if getattr(args, name) < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be nonnegative.')
    if min(args.min_size) < 32:
        raise ValueError('--min-size values must be >= 32.')
    for name in ('lr', 'momentum', 'lr_gamma'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive.')
    for name in ('weight_decay', 'norm_weight_decay', 'clip_grad', 'min_lr_ratio'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and nonnegative.')
    for name in ('warmup_factor', 'ema_decay', 'score_threshold', 'nms_threshold'):
        if not 0 < getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be in (0, 1).')
    if args.lr_schedule == 'multistep' and (
            args.lr_steps != sorted(set(args.lr_steps)) or not 0 < args.lr_steps[0]
            or args.lr_steps[-1] >= args.epochs):
        raise ValueError('--lr-steps must be strictly increasing epochs in (0, --epochs).')
    for name in ('val_max_batches', 'max_train_batches'):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive.')
    if not args.val_annotations.is_file():
        raise ValueError(f'--val-annotations not found: {args.val_annotations}')
    if args.resume == 'auto':
        candidate = args.out / 'last.pt'
        args.resume = candidate if candidate.is_file() else None
    elif args.resume is not None:
        args.resume = Path(args.resume)
        if not args.resume.is_file():
            raise ValueError(f'--resume checkpoint does not exist: {args.resume}')
    cluster = cluster_from_environment()
    if cluster.distributed and args.device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Distributed runs select the GPU from LOCAL_RANK; use --device auto.')
    return cluster


def lr_factor(step: int, *, total_steps: int, steps_per_epoch: int, warmup_iters: int,
              warmup_factor: float, schedule: str, lr_steps: list[int], gamma: float,
              min_lr_ratio: float) -> float:
    """Multiplier on the peak LR at optimizer step `step` (linear warmup x decay)."""
    warmup = 1.0
    if step < warmup_iters:
        warmup = warmup_factor + (1 - warmup_factor) * step / warmup_iters
    if schedule == 'multistep':
        decay = gamma ** sum(step >= epoch * steps_per_epoch for epoch in lr_steps)
    else:
        progress = min(step / max(1, total_steps), 1.0)
        decay = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return warmup * decay


# --------------------------------------------------------------------------- #
# Run-file publication, cluster control, W&B
# --------------------------------------------------------------------------- #
class CheckpointPublisher:
    """Rank-0 writer that never blocks training on /Volumes (FUSE) I/O.

    torch.save lands on node-local staging (fast), and a single background
    thread copies each file to its destination in submission order. A stalled
    Volume mount therefore only delays checkpoint durability -- it can never
    keep rank 0 out of the next collective past the NCCL timeout. close()
    drains the backlog and reports files that never reached their destination.
    """

    def __init__(self) -> None:
        self._staging = Path(tempfile.mkdtemp(prefix='checkpoint_staging_'))
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = {}
        self._sequence = 0
        self._buffers = {}

    def save(self, destination: Path, checkpoint: dict) -> None:
        self._sequence += 1
        staged = self._staging / f'{self._sequence:06d}_{destination.name}'
        with staged.open('wb') as file:
            torch.save(checkpoint, file)
        self._pending[destination] = self._executor.submit(self._publish, staged, destination)

    def append(self, destination: Path, line: str) -> None:
        """Append a line by rewriting the whole file through staging.

        UC Volumes (FUSE) raise OSError(29, 'Illegal seek') when opening an
        existing file in append mode, so the full content is kept in memory
        (seeded from the destination on first use, e.g. after resume).
        """
        buffer = self._buffers.get(destination)
        if buffer is None:
            buffer = self._buffers[destination] = []
            if destination.exists():
                buffer.append(destination.read_text(encoding='utf-8'))
        buffer.append(line)
        self._sequence += 1
        staged = self._staging / f'{self._sequence:06d}_{destination.name}'
        staged.write_text(''.join(buffer), encoding='utf-8')
        self._pending[destination] = self._executor.submit(self._publish, staged, destination)

    @staticmethod
    def _publish(staged: Path, destination: Path) -> None:
        temporary = destination.with_name(destination.name + '.tmp')
        try:
            shutil.copyfile(staged, temporary)
            os.replace(temporary, destination)
        finally:
            staged.unlink(missing_ok=True)

    def close(self, timeout_per_file: float = 1800) -> list[str]:
        self._executor.shutdown(wait=False)
        failures = []
        for destination, future in self._pending.items():
            try:
                future.result(timeout=timeout_per_file)
            except Exception as error:  # noqa: BLE001 -- collected for the caller
                failures.append(f'{destination}: {error!r}')
        shutil.rmtree(self._staging, ignore_errors=True)
        return failures


def _raise_keyboard_interrupt(signum, frame) -> None:  # noqa: ARG001 -- signal handler
    """Turn SIGTERM into the interrupt path so `finally` blocks still run."""
    raise KeyboardInterrupt(f'received signal {signum}')


def terminate_cluster() -> None:
    """Request termination (stop, NOT delete) of the current Databricks cluster."""
    import requests

    host = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
    token = os.environ.get('DATABRICKS_TOKEN', '')
    cluster_id = os.environ.get('DATABRICKS_CLUSTER_ID', '')
    if not (host and token and cluster_id):
        print('Cannot terminate cluster: DATABRICKS_HOST/TOKEN/CLUSTER_ID env vars are missing.',
              file=sys.stderr, flush=True)
        return
    if not host.startswith('http'):
        host = f'https://{host}'
    print(f'Requesting termination of cluster {cluster_id} ...', flush=True)
    for attempt in range(1, 6):
        try:
            response = requests.post(f'{host}/api/2.1/clusters/delete',
                                     headers={'Authorization': f'Bearer {token}'},
                                     json={'cluster_id': cluster_id}, timeout=30)
            if response.ok:
                print('Cluster termination requested; this machine will stop shortly.', flush=True)
                return
            print(f'Terminate attempt {attempt}/5 failed: HTTP {response.status_code} '
                  f'{response.text[:500]}', file=sys.stderr, flush=True)
        except requests.RequestException as error:
            print(f'Terminate attempt {attempt}/5 failed: {error}', file=sys.stderr, flush=True)
        time.sleep(10 * attempt)
    print('Failed to terminate the cluster after 5 attempts; terminate it manually.',
          file=sys.stderr, flush=True)


def wandb_init(args, config: dict, run_id: str | None):
    try:
        import wandb
    except ImportError as error:
        raise ImportError('wandb is unavailable; pip install wandb or pass --no-wandb.') from error
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_run_name or args.out.name, group=args.wandb_group,
                     mode=args.wandb_mode, tags=args.wandb_tags or None, config=config,
                     job_type='train', id=run_id, resume='allow' if run_id else None)
    # Every metric is plotted against the trainer's own step counter instead of
    # W&B's internal step, so a resumed run that replays a few steps after its
    # last checkpoint overlaps cleanly instead of having its logs rejected.
    run.define_metric('trainer/global_step')
    run.define_metric('*', step_metric='trainer/global_step')
    for name in ('val/AP', 'val/AP50', 'val/AP75'):
        run.define_metric(name, summary='max')
    run.define_metric('train/loss', summary='min')
    return run


def wandb_detection_image(image: torch.Tensor, prediction: dict, truth: dict,
                          class_names: list[str], min_score: float, canvas: tuple[int, int]):
    """wandb.Image with toggleable 'predictions' and 'ground_truth' box layers.

    The image is zero-padded bottom/right onto a shared canvas (W&B galleries
    need equal sizes); pixel-domain boxes are unaffected by that padding.
    """
    import wandb

    padded = torch.zeros((3, *canvas), dtype=image.dtype)
    padded[:, :image.shape[1], :image.shape[2]] = image
    image = padded

    labels = dict(enumerate(class_names))

    def box_data(boxes, classes, scores=None):
        rows = []
        for i, (box, cls) in enumerate(zip(boxes.tolist(), classes.tolist())):
            row = {'position': {'minX': box[0], 'minY': box[1], 'maxX': box[2], 'maxY': box[3]},
                   'domain': 'pixel', 'class_id': int(cls)}
            if scores is None:
                row['box_caption'] = class_names[cls]
            else:
                row['box_caption'] = f'{class_names[cls]} {scores[i]:.2f}'
                row['scores'] = {'score': float(scores[i])}
            rows.append(row)
        return rows

    keep = prediction['scores'] >= min_score
    return wandb.Image(image.permute(1, 2, 0).numpy(), boxes={
        'predictions': {'box_data': box_data(prediction['boxes'][keep], prediction['classes'][keep],
                                             prediction['scores'][keep]), 'class_labels': labels},
        'ground_truth': {'box_data': box_data(truth['boxes'], truth['classes']), 'class_labels': labels},
    })


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def images_to_device(images: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [image.to(device, non_blocking=True).float().div_(255) for image in images]


def build_loader(dataset, *, batch_size: int, num_workers: int, device, seed: int):
    worker_kwargs = ({'multiprocessing_context': 'spawn', 'persistent_workers': True,
                      'prefetch_factor': 4} if num_workers else {})
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=device.type == 'cuda', collate_fn=collate_detection,
                      drop_last=False, generator=torch.Generator().manual_seed(seed), **worker_kwargs)


class Trainer:
    def __init__(self, args, cluster: Cluster, device: torch.device):
        self.args, self.cluster, self.device = args, cluster, device
        self.remote, self.cache = validate_paths(str(args.mds), str(args.cache))
        self.output = args.out.resolve()
        if self.output.exists() and args.resume is None:
            raise FileExistsError(f'Run output already exists; choose a new directory or pass '
                                  f'--resume auto: {self.output}')
        if not self.output.parent.is_dir():
            raise ValueError(f'Run output parent must already exist: {self.output.parent}')
        for path in (self.remote, self.cache):
            if self.output == path or self.output in path.parents or path in self.output.parents:
                raise ValueError('Run output, dataset, and cache must be separate, non-nested directories.')

        random.seed(args.seed + cluster.rank)
        np.random.seed((args.seed + cluster.rank) % 2 ** 32)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
            # Input shapes vary per batch (aspect ratio x multi-scale), so cuDNN
            # autotuning would re-benchmark constantly; TF32 is free accuracy-wise.
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.meta = load_dataset_meta(self.remote)
        self.class_names = self.meta['class_names']
        self.category_ids = [category['id'] for category in self.meta['categories']]

        streaming_kwargs = dict(cache_limit=args.cache_limit, shuffle_seed=args.seed)
        self.train_dataset = COCODetectionDataset(
            remote=str(self.remote / 'train'), local=str(self.cache / 'train'), training=True,
            hflip_prob=args.hflip_prob, batch_size=args.batch_size, **streaming_kwargs)
        self.val_dataset = COCODetectionDataset(
            remote=str(self.remote / 'val'), local=str(self.cache / 'val'), training=False,
            batch_size=args.batch_size, **streaming_kwargs)
        loader_seed = args.seed * 1000 + cluster.rank
        self.train_loader = build_loader(self.train_dataset, batch_size=args.batch_size,
                                         num_workers=args.num_workers, device=device, seed=loader_seed)
        self.val_loader = build_loader(self.val_dataset, batch_size=args.batch_size,
                                       num_workers=max(1, args.num_workers // 2) if args.num_workers else 0,
                                       device=device, seed=loader_seed)

        if args.pretrained_backbone:  # one download, not world_size racing ones
            if cluster.primary:
                prefetch_backbone_weights(args.model)
            cluster.barrier()
        model = build_model(args.model, num_classes=len(self.class_names),
                            pretrained_backbone=args.pretrained_backbone,
                            trainable_backbone_layers=args.trainable_backbone_layers,
                            min_size=tuple(args.min_size), max_size=args.max_size,
                            detections_per_image=args.detections_per_image,
                            score_threshold=args.score_threshold, nms_threshold=args.nms_threshold)
        if cluster.distributed and args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.raw_model = model.to(device)
        self.parameters = sum(p.numel() for p in self.raw_model.parameters())
        self.trainable_parameters = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)
        self.model = (DistributedDataParallel(self.raw_model, device_ids=[device.index]
                                              if device.type == 'cuda' else None)
                      if cluster.distributed else self.raw_model)

        self.global_batch = args.batch_size * args.grad_accum * cluster.world_size
        self.learning_rate = args.lr * self.global_batch / REFERENCE_BATCH
        groups = parameter_groups(self.raw_model, args.weight_decay, args.norm_weight_decay)
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(groups, lr=self.learning_rate, momentum=args.momentum)
        else:
            self.optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
        self.batches_per_epoch = (len(self.train_loader) if args.max_train_batches is None
                                  else min(len(self.train_loader), args.max_train_batches))
        self.steps_per_epoch = max(1, math.ceil(self.batches_per_epoch / args.grad_accum))
        self.total_steps = self.steps_per_epoch * args.epochs
        schedule = dict(total_steps=self.total_steps, steps_per_epoch=self.steps_per_epoch,
                        warmup_iters=args.warmup_iters, warmup_factor=args.warmup_factor,
                        schedule=args.lr_schedule, lr_steps=args.lr_steps, gamma=args.lr_gamma,
                        min_lr_ratio=args.min_lr_ratio)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: lr_factor(step, **schedule))
        self.amp = args.amp and device.type == 'cuda'
        self.amp_dtype = (torch.bfloat16 if self.amp and torch.cuda.is_bf16_supported()
                          else torch.float16)
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp and self.amp_dtype == torch.float16)
        self.ema = ModelEMA(self.raw_model, decay=args.ema_decay) if args.ema else None

        self.start_epoch, self.skip_batches = 1, 0
        self.global_step, self.best_ap, self.best_epoch = 0, -math.inf, None
        self.skipped_steps = 0
        self.wandb_run_id = None
        if args.resume is not None:
            self._load_checkpoint(args.resume)

        self.config = self._build_config()
        self.run = None
        self.publisher = None
        self.coco_gt = None

    # ------------------------------------------------------------------ #
    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint['config']['arguments']['model'] != self.args.model:
            raise ValueError(f'--resume checkpoint was trained with a different --model: {path}')
        self.raw_model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.ema is not None:
            if checkpoint.get('ema_state'):
                self.ema.load_state_dict(checkpoint['ema_state'])
            else:
                self.ema = ModelEMA(self.raw_model, decay=self.args.ema_decay)
        self.global_step = checkpoint['global_step']
        self.best_ap = checkpoint.get('best_ap', -math.inf)
        self.best_epoch = checkpoint.get('best_epoch')
        self.skipped_steps = checkpoint.get('skipped_steps', 0)
        self.wandb_run_id = checkpoint.get('wandb_run_id')
        if checkpoint['epoch_complete']:
            self.start_epoch, self.skip_batches = checkpoint['epoch'] + 1, 0
        else:
            self.start_epoch, self.skip_batches = checkpoint['epoch'], checkpoint['batches_done']
        if self.skip_batches >= self.batches_per_epoch:
            self.start_epoch, self.skip_batches = self.start_epoch + 1, 0
        # Position the stream exactly: streaming epochs are 0-based, and the
        # sample offset is global (all ranks) -- same as StreamingDataLoader.
        self.train_dataset.load_state_dict({
            'epoch': self.start_epoch - 1,
            'sample_in_epoch': self.skip_batches * self.args.batch_size * self.cluster.world_size,
            'num_canonical_nodes': self.cluster.num_nodes,
            'shuffle_seed': self.args.seed,
            'initial_physical_nodes': self.cluster.num_nodes,
        })
        if self.cluster.primary:
            print(json.dumps({'event': 'resume', 'checkpoint': str(path), 'epoch': self.start_epoch,
                              'skip_batches': self.skip_batches, 'global_step': self.global_step}),
                  flush=True)

    def _build_config(self) -> dict:
        args = self.args
        return {
            'model': args.model,
            'arguments': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            'dataset': {'path': str(self.remote), 'splits': self.meta['splits'],
                        'created_utc': self.meta.get('created_utc')},
            'class_names': self.class_names,
            'category_ids': self.category_ids,
            'num_classes': len(self.class_names),
            'parameters': self.parameters,
            'trainable_parameters': self.trainable_parameters,
            'torch_version': str(torch.__version__),
            'device': str(self.device),
            'gpu_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
            'world_size': self.cluster.world_size,
            'effective_batch': self.global_batch,
            'learning_rate_scaled': self.learning_rate,
            'amp_dtype': str(self.amp_dtype) if self.amp else None,
            'batches_per_epoch': self.batches_per_epoch,
            'steps_per_epoch': self.steps_per_epoch,
            'total_steps': self.total_steps,
            'train_samples_per_rank': len(self.train_dataset),
            'val_samples_per_rank': len(self.val_dataset),
        }

    # ------------------------------------------------------------------ #
    def checkpoint_state(self, epoch: int, epoch_complete: bool, batches_done: int, metrics: dict) -> dict:
        return {
            'model_state': self.raw_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'ema_state': self.ema.state_dict() if self.ema is not None else None,
            'epoch': epoch, 'epoch_complete': epoch_complete, 'batches_done': batches_done,
            'global_step': self.global_step, 'best_ap': self.best_ap, 'best_epoch': self.best_epoch,
            'skipped_steps': self.skipped_steps, 'wandb_run_id': self.wandb_run_id,
            'metrics': metrics, 'config': self.config,
        }

    def log(self, values: dict) -> None:
        if self.run is not None:
            self.run.log({'trainer/global_step': self.global_step, **values})

    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> dict:
        args, cluster, device = self.args, self.cluster, self.device
        self.model.train()
        start_batch, total = self.skip_batches, self.batches_per_epoch
        self.skip_batches = 0
        if start_batch >= total:
            raise ValueError('Nothing left to train in this epoch.')
        epoch_sums = torch.zeros(len(LOSS_KEYS) + 1, dtype=torch.float64, device=device)
        interval_sums = torch.zeros_like(epoch_sums)
        epoch_batches = interval_batches = 0
        interval_data = interval_compute = 0.0
        consecutive_skips, grad_norm = 0, float('nan')
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        self.optimizer.zero_grad(set_to_none=True)
        iterator = iter(self.train_loader)
        progress = tqdm(total=total, initial=start_batch, desc=f'Epoch {epoch}/{args.epochs} train',
                        unit='batch', file=sys.stderr, mininterval=5.0, dynamic_ncols=True,
                        disable=args.no_progress or not cluster.primary)
        with progress:
            for index in range(start_batch, total):
                fetch_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:  # identical on every rank: streaming equalises partitions
                    break
                compute_start = time.perf_counter()
                images = images_to_device(batch['images'], device)
                targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()}
                           for target in batch['targets']]
                boundary = (index + 1) % args.grad_accum == 0 or index + 1 == total
                accumulating = not boundary and cluster.distributed
                with self.model.no_sync() if accumulating else contextlib.nullcontext():
                    with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=self.amp):
                        losses = self.model(images, targets)
                    components = torch.stack([losses[key].float() for key in LOSS_KEYS])
                    loss = components.sum()
                    # Always backpropagate, even a nonfinite loss: every rank must
                    # enter the gradient all-reduce. The NaN then shows up in the
                    # (identical, all-reduced) grad norm and all ranks skip together.
                    self.scaler.scale(loss / args.grad_accum).backward()
                values = torch.cat([loss.detach()[None], components.detach()]).double()
                values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                epoch_sums += values
                interval_sums += values
                epoch_batches += 1
                interval_batches += 1
                if boundary:
                    self.scaler.unscale_(self.optimizer)
                    norm = nn.utils.clip_grad_norm_(self.raw_model.parameters(),
                                                    args.clip_grad if args.clip_grad else float('inf'))
                    grad_norm = float(norm)
                    if self.scaler.is_enabled():
                        previous_scale = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        stepped = self.scaler.get_scale() >= previous_scale
                    else:
                        stepped = math.isfinite(grad_norm)
                        if stepped:
                            self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if stepped:
                        self.scheduler.step()
                        self.global_step += 1
                        consecutive_skips = 0
                        if self.ema is not None:
                            self.ema.update(self.raw_model)
                    else:
                        self.skipped_steps += 1
                        consecutive_skips += 1
                        if consecutive_skips > MAX_CONSECUTIVE_SKIPS:
                            raise RuntimeError(f'{consecutive_skips} consecutive nonfinite gradient '
                                               'steps; lower --lr or enable --clip-grad.')
                interval_data += compute_start - fetch_start
                interval_compute += time.perf_counter() - compute_start
                progress.update()

                if boundary and stepped and self.global_step % args.log_every == 0:
                    means = (cluster.sum_(interval_sums.clone())
                             / (interval_batches * cluster.world_size)).tolist()
                    elapsed = interval_data + interval_compute
                    record = {
                        'train/loss': means[0],
                        **{f'train/{key}': value for key, value in zip(LOSS_KEYS, means[1:])},
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/grad_norm': grad_norm,
                        'train/skipped_steps': self.skipped_steps,
                        'perf/images_per_second': interval_batches * args.batch_size
                                                  * cluster.world_size / max(elapsed, 1e-9),
                        'perf/data_wait_fraction': interval_data / max(elapsed, 1e-9),
                        'perf/step_time_ms': 1000 * elapsed / interval_batches,
                        'trainer/epoch': epoch - 1 + (index + 1) / total,
                    }
                    if self.scaler.is_enabled():
                        record['train/amp_scale'] = self.scaler.get_scale()
                    if device.type == 'cuda':
                        record['perf/gpu_mem_peak_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30
                        torch.cuda.reset_peak_memory_stats(device)
                    progress.set_postfix(loss=f'{means[0]:.4f}', lr=f'{record["train/lr"]:.2e}', refresh=False)
                    if cluster.primary:
                        self.log(record)
                        if args.no_progress:
                            remaining = (total - index - 1) * elapsed / interval_batches
                            print(json.dumps({'event': 'step', 'epoch': epoch, 'batch': index + 1,
                                              'of': total, 'step': self.global_step,
                                              'loss': round(means[0], 4), 'lr': record['train/lr'],
                                              'img_s': round(record['perf/images_per_second'], 1),
                                              'epoch_eta_min': round(remaining / 60, 1)}), flush=True)
                    interval_sums.zero_()
                    interval_batches, interval_data, interval_compute = 0, 0.0, 0.0

                if (boundary and stepped and args.checkpoint_every
                        and self.global_step % args.checkpoint_every == 0 and index + 1 < total):
                    if cluster.primary:
                        self.publisher.save(self.output / 'last.pt', self.checkpoint_state(
                            epoch, False, index + 1, {'epoch': epoch, 'batch': index + 1}))

        epoch_sums = cluster.sum_(epoch_sums)
        count = cluster.sum_(torch.tensor([epoch_batches], dtype=torch.float64, device=device)).item()
        means = (epoch_sums / max(count, 1)).tolist()
        return {'loss': means[0], **dict(zip(LOSS_KEYS, means[1:])), 'batches': epoch_batches,
                'elapsed_seconds': time.perf_counter() - epoch_start}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict | None:
        args, cluster, device = self.args, self.cluster, self.device
        model = self.ema.module if self.ema is not None else self.raw_model
        model.eval()
        batches = (len(self.val_loader) if args.val_max_batches is None
                   else min(len(self.val_loader), args.val_max_batches))
        rows, samples, start = [], [], time.perf_counter()
        progress = tqdm(total=batches, desc=f'Epoch {epoch}/{args.epochs} val', unit='batch',
                        file=sys.stderr, mininterval=5.0, dynamic_ncols=True,
                        disable=args.no_progress or not cluster.primary)
        with progress:
            iterator = iter(self.val_loader)
            for _ in range(batches):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                with torch.autocast(device_type=device.type, dtype=self.amp_dtype, enabled=self.amp):
                    outputs = model(images_to_device(batch['images'], device))
                for image, target, image_id, output in zip(batch['images'], batch['targets'],
                                                           batch['image_ids'], outputs):
                    prediction = {'image_id': image_id,
                                  'boxes': output['boxes'].float().cpu().numpy(),
                                  'scores': output['scores'].float().cpu().numpy(),
                                  'classes': (output['labels'] - 1).cpu().numpy()}
                    rows.append(prediction)
                    if cluster.primary and len(samples) < args.wandb_num_images:
                        samples.append((image, prediction, {'boxes': target['boxes'].numpy(),
                                                            'classes': target['labels'].numpy() - 1}))
                progress.update()
        unique = {}
        for row in cluster.gather(rows):  # streaming pads val partitions with repeats
            unique.setdefault(row['image_id'], row)
        if not cluster.primary:
            return None
        inference_seconds = time.perf_counter() - start
        metrics = evaluate_coco(self.coco_gt, list(unique.values()), list(unique),
                                self.category_ids, self.class_names)
        metrics['inference_seconds'] = inference_seconds
        metrics['eval_seconds'] = time.perf_counter() - start
        metrics['samples'] = samples
        return metrics

    # ------------------------------------------------------------------ #
    def fit(self) -> list[dict]:
        args, cluster = self.args, self.cluster
        self.cluster.barrier()
        if cluster.primary:
            self.output.mkdir(exist_ok=args.resume is not None)
            with (self.output / 'config.json').open('w', encoding='utf-8') as file:
                json.dump(self.config, file, indent=2, allow_nan=False)
                file.write('\n')
            print(json.dumps({'event': 'setup', **self.config}, allow_nan=False), flush=True)
            local_gt = Path(tempfile.mkdtemp(prefix='coco_gt_')) / args.val_annotations.name
            shutil.copyfile(args.val_annotations, local_gt)  # one bulk FUSE read
            self.coco_gt = load_coco_gt(local_gt)
            shutil.rmtree(local_gt.parent, ignore_errors=True)
            if args.wandb:
                self.run = wandb_init(args, self.config, self.wandb_run_id)
                self.wandb_run_id = self.run.id
                self.run.summary['parameters'] = self.parameters
            self.publisher = CheckpointPublisher()
        self.cluster.barrier()

        history, failures, completed = [], [], False
        try:
            for epoch in range(self.start_epoch, args.epochs + 1):
                training = self.train_epoch(epoch)
                result = {'epoch': epoch, 'global_step': self.global_step,
                          'lr': self.optimizer.param_groups[0]['lr'],
                          **{f'train_{key}': value for key, value in training.items()}}
                validate_now = epoch % args.val_every == 0 or epoch == args.epochs
                improved, metrics = False, None
                if validate_now:
                    metrics = self.evaluate(epoch)
                if cluster.primary and metrics is not None:
                    result.update({f'val_{name}': metrics[name] for name in STAT_NAMES})
                    result.update(val_per_class_ap={name: ap if math.isfinite(ap) else None
                                                    for name, ap in metrics['per_class_ap'].items()},
                                  val_eval_seconds=metrics['eval_seconds'],
                                  val_images=metrics['num_images'])
                    improved = metrics['AP'] > self.best_ap
                    if improved:
                        self.best_ap, self.best_epoch = metrics['AP'], epoch
                    print(metrics['summary'], flush=True)
                if cluster.primary:
                    result['best_ap'] = self.best_ap if math.isfinite(self.best_ap) else None
                    self._publish_epoch(epoch, result, improved)
                    self._log_epoch(epoch, result, metrics)
                history.append(result)
                cluster.barrier()
            completed = True
        finally:
            if self.publisher is not None:
                failures = self.publisher.close()
                for failure in failures:
                    print(f'Checkpoint publication failed: {failure}', file=sys.stderr, flush=True)
            # Close an interrupted run on W&B with the progress it did make,
            # instead of leaving it dangling as "crashed" with no summary.
            if not completed and self.run is not None:
                if math.isfinite(self.best_ap):
                    self.run.summary['best_ap'] = self.best_ap
                self.run.finish(exit_code=1)
        if failures:
            raise RuntimeError(f'{len(failures)} run file(s) never reached {self.output}; see stderr above.')
        if cluster.primary:
            with (self.output / '_TRAINING_SUCCESS').open('w', encoding='utf-8') as file:
                json.dump({'epochs': args.epochs, 'best_ap': self.best_ap, 'best_epoch': self.best_epoch,
                           'global_step': self.global_step, 'world_size': cluster.world_size},
                          file, allow_nan=False)
                file.write('\n')
            if self.run is not None:
                self.run.summary.update({'best_ap': self.best_ap, 'best_epoch': self.best_epoch})
                if args.wandb_log_model and (self.output / 'best.pt').is_file():
                    import wandb
                    artifact = wandb.Artifact(f'{self.run.name}-best', type='model',
                                              metadata={'best_ap': self.best_ap, 'epoch': self.best_epoch,
                                                        'model': args.model})
                    artifact.add_file(str(self.output / 'best.pt'))
                    self.run.log_artifact(artifact, aliases=['best', f'epoch-{self.best_epoch}'])
                self.run.finish()
        return history

    def _publish_epoch(self, epoch: int, result: dict, improved: bool) -> None:
        eval_model = self.ema.module if self.ema is not None else self.raw_model
        weights = {'model_state': eval_model.state_dict(), 'epoch': epoch, 'ema': self.ema is not None,
                   'metrics': result, 'config': self.config}
        if self.args.save_epochs:
            self.publisher.save(self.output / f'epoch_{epoch:03d}.pt', weights)
        if improved:
            self.publisher.save(self.output / 'best.pt', weights)
        self.publisher.save(self.output / 'last.pt', self.checkpoint_state(epoch, True, 0, result))
        self.publisher.append(self.output / 'metrics.jsonl', json.dumps(result) + '\n')
        print(json.dumps({'event': 'epoch', **{k: v for k, v in result.items()
                                               if k != 'val_per_class_ap'}}), flush=True)

    def _log_epoch(self, epoch: int, result: dict, metrics: dict | None) -> None:
        if self.run is None:
            return
        import wandb

        logged = {'trainer/epoch': epoch,
                  'epoch/train_loss': result['train_loss'],
                  **{f'epoch/train_{key}': result[f'train_{key}'] for key in LOSS_KEYS},
                  'epoch/elapsed_minutes': result['train_elapsed_seconds'] / 60}
        if metrics is not None:
            logged.update({f'val/{name}': metrics[name] for name in STAT_NAMES})
            logged.update({'val/best_AP': self.best_ap, 'val/eval_seconds': metrics['eval_seconds'],
                           'val/num_detections': metrics['num_detections']})
            logged.update({f'val_per_class/AP_{name}': ap for name, ap in metrics['per_class_ap'].items()
                           if math.isfinite(ap)})
            table = wandb.Table(columns=['class', 'AP', 'AP50'], data=[
                [name, metrics['per_class_ap'][name], metrics['per_class_ap50'][name]]
                for name in self.class_names])
            logged['val/per_class_table'] = table
            logged['val/per_class_AP_chart'] = wandb.plot.bar(table, 'class', 'AP',
                                                              title=f'Per-class AP (epoch {epoch})')
            if metrics['samples']:
                canvas = (max(image.shape[1] for image, _, _ in metrics['samples']),
                          max(image.shape[2] for image, _, _ in metrics['samples']))
                logged['val/predictions'] = [
                    wandb_detection_image(image, prediction, truth, self.class_names,
                                          self.args.wandb_image_score, canvas)
                    for image, prediction, truth in metrics['samples']]
        self.log(logged)
        self.run.summary.update({'best_ap': self.best_ap if math.isfinite(self.best_ap) else None,
                                 'best_epoch': self.best_epoch})


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def train(args) -> list[dict]:
    cluster = validate_args(args)
    device = resolve_device(cluster, args.device)
    if cluster.distributed:
        # A wedged /Volumes mount once held rank 0 in write I/O past NCCL's
        # 10-minute default (Lyft run); COCOeval on rank 0 also takes ~1 min.
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo',
                                device_id=device if device.type == 'cuda' else None,
                                timeout=timedelta(hours=2))
    try:
        return Trainer(args, cluster, device).fit()
    finally:
        if cluster.distributed and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Python kills the interpreter on SIGTERM without unwinding, so a signalled
    # run would drop staged checkpoints and leave W&B dangling. torchrun relays
    # SIGTERM to every rank, so all ranks unwind together.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        train(args)
    except KeyboardInterrupt:
        parser.exit(130, 'Training interrupted. Checkpoints so far remain in the run directory; '
                         'relaunch with --resume auto.\n')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'Training failed: {error}\nRelaunch with --resume auto once fixed.\n')


if __name__ == '__main__':
    main()
