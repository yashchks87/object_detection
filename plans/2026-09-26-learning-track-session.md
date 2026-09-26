# Session log: learning track (write the model yourself, AI as tutor)

Date: 2026-09-26
Chat transcript (condensed but complete) between Yash and Devin covering a
training status check, COCO SOTA / TTA / training-cost questions, the decision
to hand-write the model for learning, and the setup of a reusable "learning
track" workflow. Part A is project-agnostic and meant to be copied into future
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
- Yash committed and pushed: `28be3ca Learning track is added.`

---

## Current state (end of session, 2026-09-26 ~20:45 UTC)

- **Running:** `frcnn_v2_001` epoch 22/26 (batch ~5,300/7,393 at 20:46 UTC),
  best val AP 0.3984 (epoch 21). ~4 epochs left (~4 h); the cluster auto-stops
  after the run (`--terminate-cluster`).
- **Learning track:** step 1 (box ops) ready for Yash to implement; nothing
  implemented yet.

```bash
cd /root/object_detection && source venv/bin/activate
python -m pytest scripts/my_frcnn -q -p no:warnings       # learning-track tests (CPU)
tail -f /local_disk0/run_logs/frcnn_v2_001.log            # monitor training
```

- **Uncommitted:** this plan file only.
- **Next steps:** Yash implements `box_ops.py` until 21/21 pass, then asks for
  review and "step 2" (NMS). After the run ends: fill in the Results section of
  `plans/exp001-frcnn-v2-baseline.md` (final AP is unverified until then).
