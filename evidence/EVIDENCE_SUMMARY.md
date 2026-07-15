# Evidence Pack — TFLite Coordinate-Normalization Root Cause

**Question:** is the YOLOv8 on-device accuracy gap caused by the `.pt → .tflite` conversion, or by how the exported model is decoded on-device?

**Answer (proven below):** the exported graph is faithful. The **only** difference between `.pt` and `.tflite` is that TFLite emits bounding-box `xywh` **normalized to `[0,1]`** (divided by the 640 input size) while `.pt` emits pixels. A decoder that skips the `× 640` step collapses every box. Class scores are identical within float32 precision.

Everything here was produced by feeding the **same preprocessed image to both models**. Scripts are included (`generate_evidence.py`, `same_tensor_decode.py`, `reexport_compare.py`) so every number is reproducible.

---

## What was tested (`provenance.json`)

| | |
|---|---|
| Source model | `Model_2026-06-27_14-44-47.pt` (sha256 `59dcac6d…`) |
| Exported model | `Model_2026-06-27_14-44-47_float32.tflite` (sha256 `cc48c597…`) |
| Images | 7 fish photos in `test_images/` |
| Toolchain | ultralytics 8.1.0, torch 2.1.2+cpu, tensorflow 2.13.1, numpy 1.24.3 (CPU / XNNPACK) |

> The test images are generic stock fish photos (not the app's exact 339 training species), so several detect only weakly — that is expected and does not affect the coordinate-normalization proof, which is about the model graph, not detection quality.

---

## Test 1 — Raw-tensor parity on identical input (`raw/parity_all.csv`, `plots/`)

Both models were given the identical preprocessed tensor; their raw `[1,343,8400]` outputs were compared element-by-element, before any decoding.

| Measurement | Result | Meaning |
|---|---|---|
| Class-score rows | max abs diff **≤ 3.6e-06** (all images) | classification is **identical** |
| Box-coordinate rows | `box_pt / box_tflite` median **= 640.0** on every image; residual after ×640 **≈ 0.002** | boxes differ **only** by ÷640 |
| Global box slope | **640.00** | see `plots/box_scatter.png` |
| Class-score R² | **1.000000** | see `plots/classscore_parity.png` |

- `plots/box_scatter.png` — TFLite box values vs `.pt` box values fall exactly on a line of slope 640.
- `plots/ratio_hist.png` — the ratio of box values spikes at 640.
- `plots/classscore_parity.png` — class scores lie on `y = x` (R² = 1.0).

## Test 2 — Same-tensor decode parity (`raw/same_tensor_decode_parity.csv`)

From the same input tensor, we ran the **full decode** (xywh→xyxy → un-letterbox → per-class NMS at conf 0.25) two ways: `.pt` boxes as-is, and TFLite boxes **× 640**. Result: **identical detections on all 7 images** (matching counts; top-1 IoU = 1.0 wherever a fish is present — Harlequin 1v1, multi-fish 2v2, species-river 10v10). Once ×640 is applied, `.pt` and TFLite are the same model.

## Test 3 — Decode demo overlays (`overlays/`) — the visual proof

For each image, three panels on the same photo: **`.pt` reference** (blue), **TFLite decoded correctly with ×640** (green), **TFLite decoded WRONG without ×640** (red).

- Correct decode reproduces the `.pt` boxes (IoU vs `.pt` ≈ 0.95–0.99 on 6/7 images).
- Wrong decode: **every box collapses to a sub-pixel box in the top-left corner** (IoU vs `.pt` = 0.0 on all images) — exactly the on-device failure.
- Best example: `overlays/overlay_species-river-fish-white-background-35038441.webp.png` (10 fish correctly boxed vs all lost).
- `overlays/decode_compare_<img>.json` has the per-image numbers.

> One image (`pexels-photo-13070712`, underwater, multiple fish) shows a low correct-vs-`.pt` top-1 IoU. That is a top-1 *ranking* difference — `.pt` and TFLite each surface a different borderline fish under Ultralytics' two preprocessing paths — not a decode error. Test 2 confirms they are identical (2 vs 2, IoU 1.0) on identical input.

## Test 4 — Model vs model, both correctly decoded (`model_vs_model/compare_all.csv`, `plots/conf_sweep.png`)

Running both models through Ultralytics' own (correct) decoder: strong detections agree tightly (`species-river` 11 vs 10 dets, top-1 IoU 0.988; `decaying-fish` 1 vs 1, IoU 0.983). Weak/borderline detections on these out-of-distribution stock photos can differ by ±1–2 near the 0.25 threshold, because the torch and tflite backends preprocess independently — this is threshold noise, not a model difference (Test 2 removes it by using one shared input). `conf_sweep.png` shows the two detection-count curves overlapping.

## Test 5 — Re-export reproducibility (`reexport/reexport_parity.json`)

We re-exported the `.pt` → `.tflite` locally with the pinned toolchain and compared our re-export against the shipped `.tflite` on identical input: **max abs diff 1.7e-06, identical `[1,343,8400]` shape**, and our re-export **also** emits normalized boxes. So the ÷640 convention is a stable, reproducible property of the Ultralytics export — the shipped binary is uncorrupted, and the `×640` fix is durable across future exports. (`reexport/reexport_float32.tflite` is included.)

---

## Conclusion & fix

1. The `.pt → .tflite` export is **faithful**; classification is identical.
2. TFLite bounding boxes are **normalized `[0,1]`**; the device decoder must **multiply xywh by 640**, then convert to corners, remove letterbox padding, and run per-class NMS.
3. This is specified end-to-end, with a validated golden sample, in **`TFLITE_CONTRACT.md`** (see the `deliverable/` bundle). No model change or re-training is required for this fix.

## Out of scope (separate tracks)
- **`data (8).yaml`** is byte-identical to `data_338.yaml` — no change to analyze. The class-list content issues (nc=339 including test/QA + duplicate-binomial entries) are a **retrain-side** concern, independent of this decode fix.
- Quantitative before/after mAP on the labeled validation set is still pending the Azure data pull.
