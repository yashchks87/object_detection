# Experiment 001: Faster R-CNN R50-FPN v2 baseline (COCO 2017)

**Run name:** `frcnn_v2_001` | **W&B:** `yashchks87/COCO object detection/frcnn_v2_001`
**Goal:** first strong, reproducible baseline on COCO val2017 using the standard
torchvision/detectron2 recipe, trained from an ImageNet backbone (no COCO
pretraining). All later experiments are compared against this run.

**Launch:** `./scripts/run_training.sh frcnn_v2_001` (all defaults below).

## Model / architecture

| Choice | Value | Why |
|---|---|---|
| Detector | Faster R-CNN, `fasterrcnn_resnet50_fpn_v2` (torchvision) | Two-stage, strong and well-understood baseline; v2 is the improved recipe (~+3-9 AP over v1 in torchvision's benchmarks, depending on schedule) |
| Backbone | ResNet-50, ImageNet-1k pretrained (`IMAGENET1K_V1`) | Standard detection init; COCO detection weights deliberately NOT used |
| Neck | FPN (P2-P6, 256 ch) with BatchNorm | v2 recipe |
| RPN head | 2 conv layers (v2), anchors 32-512 x ratios 0.5/1/2 | torchvision default |
| Box head | 4 conv + 1 FC (1024) with BatchNorm | v2 recipe |
| Classes | 80 COCO classes + background (81 logits) | |
| Frozen parts | stem + `layer1` (`trainable_backbone_layers=3`); their BN -> FrozenBatchNorm | Standard; avoids BN-stat drift in frozen stages |
| Normalisation | SyncBatchNorm across the 4 GPUs (trainable BN) | Only 4 images/GPU -> per-GPU BN stats too noisy |
| Parameters | 43.66M total, 43.44M trainable | |

## Data

| Item | Value |
|---|---|
| Train / val | COCO train2017 118,286 imgs / val2017 5,000 imgs (MDS shards) |
| Train input size | multi-scale: short side random in {640, 672, 704, 736, 768, 800}, long side <= 1333 |
| Eval input size | short side 800, long side <= 1333 |
| Augmentation | horizontal flip p=0.5 (+ multi-scale) |
| Targets | crowd boxes dropped, boxes clipped to image, degenerate (<0.01px) removed; empty images kept as negatives |

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD, momentum 0.9 |
| Base LR | 0.02 at global batch 16 (linear scaling rule; here global batch = 16, so LR = 0.02) |
| Batch size | 4 per GPU x 4 A10G = 16 global, no grad accumulation |
| Weight decay | 1e-4 (0 on normalisation-layer params) |
| Warmup | linear, 1,000 iterations, from 0.001 x LR |
| LR schedule | multistep: x0.1 after epoch 16 and after epoch 22 |
| Epochs | 26 (~2x schedule; torchvision reference length) |
| Steps | 7,393 per epoch, 192,218 total; LR drops at step 118,288 and 162,646 |
| Precision | bf16 autocast, TF32 matmuls |
| Grad clipping | off (grad norm logged) |
| EMA | off |
| Inference | score threshold 0.05, NMS IoU 0.5, max 100 detections/image |
| Seed | 42 |

## Evaluation & checkpoints

- Official COCO bbox metrics (pycocotools) on all 5,000 val images every
  epoch: AP@[.5:.95] (primary), AP50, AP75, APs/m/l, AR1/10/100, per-class AP.
- `best.pt` = highest val AP; `last.pt` every epoch + every 2,000 steps
  (exact mid-epoch resume with `--resume auto`).

## Expectations

- Throughput ~36 img/s total -> ~55 min/epoch (+~2 min eval), **~24 h total**.
- Memory ~7 GB of 23 GB per GPU.
- Reference points: torchvision v1 R50-FPN 26 epochs = 37.0 AP. With v2
  architecture + multi-scale training, expect roughly **~40-42 AP**
  (estimate, not a measured number).
- Watch in W&B: `val/AP` jumps after epochs 16 and 22 (LR drops);
  `perf/data_wait_fraction` should stay ~0 (GPU-bound).

## Ideas for next experiments (not part of exp001)

- `--ema` (weight EMA), `--lr-schedule cosine`, longer schedule (36 epochs / 3x).
- Larger-scale jitter / stronger augmentation.
- Compare classic `fasterrcnn_resnet50_fpn` (v1) as a lower reference.

## Results

_Fill in after the run:_ best val AP = ___ at epoch ___; AP50 ___; AP75 ___; APs/m/l ___ / ___ / ___.
