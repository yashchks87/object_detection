# Session log: learning track (write the model yourself, AI as tutor)

Date: 2026-09-26
Chat transcript (condensed but complete) between Yash and Devin covering a
training status check, COCO SOTA / TTA / training-cost questions, the decision
to hand-write the model for learning, the setup of a reusable "learning
track" workflow, and the first learning steps (`box_area`, `box_iou`). Part A is project-agnostic and meant to be copied into future
projects; Part B is what happened in this repo.

---

# Part A: Reusable learning-track playbook (any project)

## A1. Goal and split of responsibilities

Two goals, handled differently:

| Area | Who writes code | How Yash learns it |
|---|---|---|
| Core model / algorithm (the thing to understand deeply and explain in interviews) | **Yash only** | Implements from specs + failing tests; AI reviews and hints |
| Production scaffolding (data pipeline, distributed training, checkpointing, logging, launchers) | AI may write it | Read one file at a time, ask "why" questions, make small changes yourself, be able to whiteboard the system |

Keep the library/reference implementation as the baseline; the hand-written
version lives next to it (e.g. `--model my_<name>`), never replaces it.

## A2. Tutor-mode rules (put in the repo's `AGENTS.md`)

Rules must live in the project rules file so every future AI session follows
them, not just the current chat. Template:

```text
## Learning track: hand-written <model> (`<path>/my_<model>/`)
- NEVER write or edit implementation code in `<path>/my_<model>/` (anything that is not a test),
  and never paste working solutions into chat. If the user explicitly says "write it for me",
  confirm once before doing so.
- Allowed: specs (signatures + docstrings with `raise NotImplementedError('TODO(you): ...')`),
  failing tests (`test_*.py`, reference library as the oracle), reviewing diffs (point to the bug,
  explain why, do not fix it), explaining the reference implementation only AFTER tests pass.
- Hints only on request, escalating: (1) conceptual nudge, (2) paper/section pointer, (3) pseudocode.
- Tests: `<test command>` (CPU-only if possible).
- Build order: <numbered list of components>; next step's stubs + tests only once the current passes.
- Validation ladder: unit tests -> overfit a tiny subset -> smoke run -> short real run vs. baseline.
```

## A3. What the AI prepares for each step

1. **Stub module**: function signatures, docstrings with shapes, conventions,
   edge cases (empty inputs, dtype/device, differentiability), and a pointer to
   the paper section instead of the formula. Bodies raise
   `NotImplementedError('TODO(you): <name>')`.
2. **Failing test file**: hand-computed known values, comparison against the
   reference library (the "oracle"), invariants (symmetry, range, round trip,
   identity), edge cases, numerical safety (clamps, no inf/NaN).
3. **Validate the tests before handing them over**: temporarily monkeypatch the
   stubs with the reference implementation in a throwaway script (outside the
   repo) and confirm all tests pass; confirm they all fail on the stubs. Then a
   failing test always means a bug in the user's code, never in the test.

## A4. Validation ladder for a hand-written model

| Rung | Catches | Cost |
|---|---|---|
| Unit tests vs. reference library, per component | wrong formulas, shapes, edge cases | seconds |
| Equivalence test: same weights in both models, compare intermediate outputs | wiring / convention mismatches | seconds |
| Overfit ~10 samples to ~0 loss | broken loss, targets, gradients | minutes |
| Smoke run through the real training script | integration, logging, checkpointing | minutes |
| Short real run (1-2 epochs) vs. the baseline's metrics at the same epoch | subtle bugs that only cost accuracy | hours |

Record the baseline's per-epoch metrics up front so the short run has an exact
target (within ~1 point = almost certainly correct).

## A5. Making it stick (learning habits)

- Struggle 30-60 min before asking for a hint.
- After each component, write a short note in your own words (what, why, the
  trade-off). If you cannot explain it, you do not understand it yet.
- After ~1 week, delete the core utilities and rewrite them from a blank file
  (target: under 15 min each). This is the interview skill.
- Read the key papers alongside the code; one-page summary each
  (problem, key idea, trade-off).
- Compare your version with the reference only after your tests pass; note the
  differences.

## A6. Interview prep checklist (ML / CV roles)

| Area | Practice |
|---|---|
| Coding | Core utilities from scratch (e.g. IoU, NMS, mAP); general DS&A (LeetCode medium) |
| Model concepts | Architecture families and their trade-offs; loss functions; how the main metric is computed |
| Training | Normalisation (BN / SyncBN / GN), LR warmup + schedules, linear scaling rule, mixed precision, EMA, debugging a loss that will not go down |
| ML system design | Large-scale training pipeline; low-latency serving (batching, TensorRT, quantisation, accuracy vs. cost trade-offs) |
| Own projects | Explain every design choice and its trade-off; incident stories from session logs (what broke, diagnosis, fix) |

Ask the AI for mock interviews ("quiz me on X", "give me a system design
question") with answers withheld until you have tried.

## A7. Useful prompts during a learning track

| Prompt | Effect |
|---|---|
| "review my <module>" | Bugs pointed out, not fixed |
| "hint 1 / 2 / 3 on <function>" | Escalating hint |
| "step N" | Next step's stubs + failing tests (only after the current step passes) |
| "explain the reference version" | Walk-through of the library code (after your tests pass) |
| "quiz me on <topic>" | Interview-style questions |

## A8. Per-function teaching loop (what worked in practice)

1. **Explain in plain words first** (no implementation): what the function
   means, a worked numeric example, where it is used in the pipeline, edge
   cases, then 3-4 guiding questions that lead to the implementation.
2. **Teach a new library primitive on a toy tensor**, not on the real problem
   (e.g. `clamp`, broadcasting with `None`), so the user still has to apply it.
3. **User attempts; AI reviews** by running the tests, listing bugs in the
   order they will be hit, explaining why each is wrong, and not fixing it.
4. **Full solution only after a genuine attempt and an explicit request**,
   verified in a throwaway script first (never edit the user's file), with a
   line-by-line "how each fix works" and a shape table. The user retypes it and
   rewrites it from memory the next day.
5. **Flow-level explanation on request**: where the function is called in the
   full system, with realistic sizes, so the shapes make sense.

## A9. Generic engineering lessons from this track

- **Debug tensor code by printing shapes** (`pytest -k <test> -s`); every
  intermediate should have the shape you predicted.
- **Malformed inputs: validate at the boundaries** (data loading / target
  preparation), **clamp where negatives are part of the math** (e.g.
  intersections), **trust inside hot paths**. Never silently "repair" data
  (e.g. swapping coordinates) because it hides format bugs.
- Clamp **each factor before multiplying**: two negatives multiply to a
  plausible-looking positive.
- A behaviour that differs from the reference library on purpose must be
  documented in the docstring and pinned by a test, otherwise it can silently
  regress.
- Prefer explicit element-wise ops (`torch.maximum/minimum`) over overloaded
  ones (`torch.max(a, b)`), which read like reductions.
- Keep `requirements.txt` hand-pinned to direct dependencies; `pip freeze >`
  dumps every installed package and is hard to maintain across machines.

---

# Part B: This session (object_detection repo)

## 1. Training status check

**Request:** "How's the model training going?"

- `frcnn_v2_001` (torchvision `fasterrcnn_resnet50_fpn_v2`, COCO 2017, 4x A10G,
  26 epochs, LR drops after epochs 16 and 22) running healthy since the 02:39
  UTC resume; no errors/NaN/OOM in the log; ~37 img/s, ~55 min/epoch incl.
  ~1.5 min COCOeval; GPUs 86-93% util, 15-19 GB of 23 GB.
- Val AP (val2017): ep1 19.7, ep5 29.5, ep10 30.6, ep15 32.8, ep17 38.9
  (first LR drop: +6 AP), ep20 39.8, ep21 39.84 (plateau). Expected final
  ~40.5-41 AP after the second drop (estimate).
- `--terminate-cluster` is set: the cluster stops after the run ends.

## 2. Q&A: highest AP ever reported on COCO

- **66.0 box AP on test-dev: Co-DETR (Co-DINO) with EVA-02 ViT-L** (304M
  params, Objects365 pretraining, with TTA; 65.4 val without TTA). Previous:
  InternImage-G ~65.5 (~3B params). Focal-Stable-DINO 64.8 test-dev with public
  data only. MegDetV2 61.0 (COCO challenge 2019/2020 winner, ensemble).
  Sources: Co-DETR repo / HF model cards, SOTA2 (successor of
  paperswithcode) leaderboard. No verified higher number found (2026-09-26).
- Gap from ~40 AP to 66 comes mainly from backbone scale, large-scale
  pretraining (Objects365), DETR-style detector + strong aug, and TTA.

## 3. Q&A: test-time augmentation (TTA)

- TTA = run several transformed views (multi-scale, h-flip), map boxes back to
  original coordinates, pool, merge per class (NMS / Soft-NMS / WBF), keep
  top-100. Typically +0.5-2 AP at N x inference cost. WBF averages coordinates
  (score-weighted) and helps AP75 most.
- Verified in code that this repo uses **no TTA**: `hflip_prob` is 0 at eval
  (`scripts/coco_data.py`), eval uses `min_size[-1]` = 800 only, one forward
  pass per batch in `evaluate()` (`scripts/train_frcnn.py`). The only merging is
  the model's internal NMS (IoU 0.5, max 100 dets).
- **Decision:** not adding TTA now. Baseline must stay a clean single-scale
  number (all experiments compare against it); ~5x longer per-epoch val; no
  lasting model gain. If ever needed: a standalone `scripts/eval_tta.py` on
  `best.pt` reporting AP with and without TTA.

## 4. Q&A: why ~24 h of training

- Mostly total compute, not the two-stage design: 118,286 imgs x 26 epochs
  = ~3.1M image passes at ~1 MP resolution; v2 model ~280 GFLOPs/img at
  inference (v1 ~134); A10G is a mid-range GPU; run is GPU-bound (86-93% util).
- Two-stage overhead (RoIAlign on 512 proposals/img, box head, RPN matching,
  NMS) is real but secondary; one-stage (RetinaNet) costs about the same, YOLO
  is cheaper per image but trains 300+ epochs.
- Speed levers: faster/more GPUs > 12-epoch schedule (~-2-3 AP) > v1 model >
  lower resolution > larger batch.

## 5. Decision: hand-write the model for learning

**Request:** reimplement Faster R-CNN instead of using torchvision?

- Recommendation given: not for results (silent AP-costing bugs, 24 h per full
  check, torchvision already validated; the custom production code is the
  valuable part), but yes for learning, alongside torchvision.
- **Yash's decision:** goal is learning + interview prep, not production. He
  writes the model code himself and does a quick 2-epoch run; the production
  scaffolding stays AI-written and is studied as a system.
- Advice given: Part A of this file (tutor mode, build order, validation
  ladder, habits, interview checklist).

## 6. Setup of the learning track

| File | Purpose |
|---|---|
| `AGENTS.md` (section "Learning track") | Tutor-mode rules, build order, test command, validation ladder, 2-epoch targets |
| `scripts/my_frcnn/box_ops.py` | Step 1 stubs: `box_area`, `box_iou`, `clip_boxes_to_image`, `remove_small_boxes`, `encode_boxes`, `decode_boxes` (xyxy, `(h, w)` sizes, weights `(10, 10, 5, 5)`, clamp `log(1000/16)`) |
| `scripts/my_frcnn/test_box_ops.py` | 21 tests; oracle = `torchvision.ops` + `torchvision.models.detection._utils.BoxCoder` |

- Build order: 1 box ops, 2 NMS, 3 FPN, 4 anchor generator, 5 matcher +
  balanced sampler, 6 RPN, 7 RoIAlign (torchvision op first, own later),
  8 box head, 9 wire in as `--model my_frcnn` in `train_frcnn.py`.
- 2-epoch targets (torchvision baseline `frcnn_v2_001`): epoch 1 loss 0.794 /
  19.7 AP, epoch 2 loss 0.650 / 25.5 AP.
- Verification: tests validated with torchvision monkeypatched in (21/21
  pass, via a throwaway script in `/tmp`, deleted); on the stubs 21/21 fail
  with `NotImplementedError` (Yash confirmed by running them).
- Yash committed and pushed: `28be3ca Learning track is added.`; this plan
  file: `ba838f8`.

## 7. `box_area`: concept, implementation, malformed boxes

- Explained in plain words: xyxy, image origin top-left with y pointing down,
  continuous coordinates (no VOC-style `+1`), degenerate boxes (area 0), and
  where area is used (IoU union, COCO small/medium/large buckets at 32² / 96²,
  FPN level assignment by sqrt(area), filtering).
- Yash wrote it himself (column slicing `boxes[:, k]`, vectorised); 3/3 tests
  passed on the first try. Review: correct; optional `boxes[..., k]` for
  batched inputs.
- **Correction by Devin:** the hint "malformed boxes come from `decode_boxes`"
  was wrong; the log-space parametrisation (`w = w_a * exp(dw)`) always gives
  positive sizes. Real sources: format mix-ups (COCO JSON is xywh, TF uses
  yxyx, DETR cxcywh), flip bugs, annotation errors, heads that regress corners
  directly.
- Options compared: trust (torchvision `box_area`), clamp, validate and raise
  (torchvision Faster R-CNN checks training targets), repair (rejected).
- Yash asked for the PyTorch clamp: taught on a toy tensor (`t.clamp(min=0)`,
  no in-place `clamp_`, zero gradient where clamped, `relu` equivalent); Yash
  then added the clamp himself **per width and height before multiplying**.
  Verified: (30,10,10,40) -> 0, (30,40,10,10) -> 0 (not a fake 600),
  malformed boxes get zero gradient. This intentionally differs from
  `torchvision.ops.box_area` (returns -600 / +600).
- Open polish (not done): docstring note "malformed boxes get area 0", a
  `test_malformed_boxes_have_zero_area` written by Yash, remove the commented
  `raise`, PEP 8 `clamp(min=0)`.

## 8. `box_iou`: concept, attempt, review, solution

- Explained: IoU = overlap / union, why union (fair to both boxes), uses (RPN
  0.7/0.3, RoI head 0.5, NMS 0.5, AP50/AP75/AP@[.5:.95]), the overlap rectangle
  (larger x1/y1, smaller x2/y2), inclusion-exclusion union, worked test cases,
  and why the clamp from `box_area` returns for disjoint boxes; broadcasting
  taught on a toy tensor (`a[:, None] + b[None, :]`, `torch.maximum/minimum`).
- Yash's attempt had the right logic (max/min edges, reusing `box_area`,
  inclusion-exclusion) but 4 bugs, reviewed without fixing: (1) `x2` assigned
  twice so `y2` undefined; (2) no broadcasting, [N] vs [M] compared
  element-wise (crash if N != M, silent diagonal if N == M); (3)
  `box_area(torch.stack(...), dim=1)`: `dim` passed to the wrong function and
  stacked [N, M] tensors break `boxes[:, k]` indexing; (4) `area1 + area2` not
  broadcast. Debugging tip: print shapes, run one test with `-s`.
- Yash fixed (1), then explicitly asked for working code. Devin verified the
  solution in a throwaway `/tmp` script (8/8 area+IoU tests) and gave it in
  chat with per-fix explanations and a shape table
  (`boxes1[:, None, k]` -> [N, 1], `boxes2[None, :, k]` -> [1, M]; inline
  clamped intersection instead of `stack`; `area1[:, None] + area2[None, :]`).
  Yash pasted it into `box_ops.py`; 5/5 IoU tests pass.
- Interview follow-ups noted: zero-area pairs give 0/0 = NaN (epsilon fix);
  memory of [N, M] intermediates (~80 MB per float32 tensor for 200k x 100).

## 9. Flow: why `box_iou` returns [N, M]

Yash asked whether the shapes differ because "1 is GT and the rest are
proposals". Answer: it is **all** GT boxes x **all** anchors/proposals; the
"1 vs K" case is NMS. Call sites in one training step (800x1216 image, ~7 GT):

| Call | Inputs | Matrix | Used for |
|---|---|---|---|
| RPN matching | ~243k anchors (P2 182,400 / P3 45,600 / P4 11,400 / P5 2,850 / P6 741; 3 ratios per cell) vs GT | [243k x 7] | row max -> obj (>= 0.7) / bg (< 0.3) / ignore; column max -> every GT keeps its best anchor (low-quality matches) |
| RPN NMS | kept proposal vs remaining | [1 x K] | remove duplicate proposals (-> ~2,000) |
| RoI head matching | ~2,000 proposals + appended GT vs GT | [2,007 x 7] | fg (>= 0.5) with class / bg; sample 512 (25% fg) |
| Evaluation | <= 100 detections vs GT | [100 x 7] | COCOeval hits at IoU 0.50 ... 0.95 |

- torchvision's matcher calls `box_iou(gt, anchors)` (GT as rows); keep one
  orientation in step 5 or the max is taken over the wrong dimension.

## 10. Repo housekeeping during the session

- Yash ran `pip freeze > requirements.txt`: the 10 hand-pinned packages were
  replaced by 129 lines of everything installed. Devin flagged it
  (`git checkout requirements.txt` to restore), but it was committed in
  `c06a7db Learning continued.` together with `box_ops.py` and pushed.
  Still open: restore the pinned version from `ba838f8`
  (`git checkout ba838f8 -- requirements.txt`) and add any genuinely new
  dependency by hand.

---

## Current state (end of session, 2026-09-26 ~22:56 UTC)

- **Training:** `frcnn_v2_001` epoch 25/26 in progress (batch ~430/7,393 at
  22:56 UTC, ~50 min/epoch left, i.e. ~2 epochs ≈ 1.8 h). LR 0.0002 since
  epoch 23. Val AP: ep22 39.99, **ep23 40.39** (second LR drop +0.4),
  ep24 40.41 (AP50 61.45, AP75 43.57, APs/m/l 23.5 / 44.3 / 52.6). Final AP
  unverified until the run ends; the cluster then auto-stops
  (`--terminate-cluster`).
- **Learning track:** `box_area` (Yash, with clamp) and `box_iou` (solution
  given after his attempt) done: **8/21 tests pass**; the 13 failures are the
  remaining stubs (`clip_boxes_to_image`, `remove_small_boxes`,
  `encode_boxes`, `decode_boxes`).

```bash
cd /root/object_detection && source venv/bin/activate
python -m pytest scripts/my_frcnn -q -p no:warnings                  # all learning-track tests (CPU)
python -m pytest scripts/my_frcnn -q -p no:warnings -k ClipAndFilter # next functions
tail -f /local_disk0/run_logs/frcnn_v2_001.log                       # monitor training
```

- **Committed/pushed:** everything through `c06a7db`. **Uncommitted:** this
  plan update.
- **Next steps:**
  1. Yash implements `clip_boxes_to_image` and `remove_small_boxes` without
     hints (hint given: `remove_small_boxes` returns int64 indices; find the
     PyTorch function returning positions where a boolean mask is True).
  2. Then `encode_boxes` / `decode_boxes` (read Faster R-CNN Sec. 3.1.2
     first), then "step 2" (NMS).
  3. Tomorrow: rewrite `box_iou` from memory (target < 5 min).
  4. `box_ops.py` polish from section 7; remove commented `raise` lines and
     trailing whitespace in `box_iou`.
  5. Restore the pinned `requirements.txt` (section 10).
  6. After the run ends: fill in the Results section of
     `plans/exp001-frcnn-v2-baseline.md`.
