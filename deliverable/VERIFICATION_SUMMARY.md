# Verification Summary — YOLOv8 `.pt` → `.tflite` Export

**Model:** `Model_2026-06-27_14-44-47_float32.tflite` (FP32, non-quantized), exported from `Model_2026-06-27_14-44-47.pt` (Ultralytics YOLOv8n, 339 classes).

**Purpose:** determine whether the reported on-device accuracy gap originates in the `.pt → .tflite` conversion or in how the exported model is consumed on-device.

**Conclusion:** the exported graph is faithful to the source model. The difference between the two is a single, well-defined and fully recoverable **box-coordinate convention** — not a loss of model quality. The `TFLITE_CONTRACT.md` document specifies exactly how to consume the model correctly.

---

## Method

Both models were given the **identical preprocessed input tensor** (aspect-preserving letterbox to 640×640, RGB, values scaled to `[0,1]`). We then compared their **raw output tensors** element-by-element, before any decoding or non-max suppression. This isolates the model graph itself from any downstream decode logic.

Environment: CPU inference, using the same library versions as the production export toolchain.

## Measurements

| Output component | Result |
|---|---|
| **Class scores** (339 per prediction) | **Identical within float32 precision** — maximum absolute difference **`7.7e-07`** across the full output. |
| **Box coordinates** (`cx, cy, w, h`) | Differ by a single constant factor: the TFLite values are the pixel values **divided by 640** (the input size). Median ratio of `.pt` to `.tflite` box values: **exactly `640.0000`**; residual after rescaling: `0.0015`. |

**Interpretation:** the TFLite export emits bounding boxes **normalized to `[0,1]`**, whereas the source model emits them in **pixel units (0–640)**. This is a standard, documented Ultralytics export convention. Any consumer must multiply the box outputs by the input size (640) before decoding. Classification is unaffected.

## Hypotheses tested

| # | Hypothesis | Verdict |
|---|---|---|
| H1 | Output tensor layout is transposed by the converter (`[1,8400,343]` vs `[1,343,8400]`) | **Not present** — output is `[1,343,8400]` (attributes-first), matching the source model. |
| H2 | The export/conversion toolchain corrupts the graph | **Ruled out** — class scores identical within float32 precision; boxes differ only by the deterministic `/640` factor. |
| H3 | Box-coordinate convention mismatch (normalization) | **Confirmed** — this is the cause; see measurements above. |
| H4 | Class-list ordering mismatch | **Not an export issue** — the model's embedded class list, the shipped names file, and the dataset class definition are mutually consistent (339 classes, same order). |
| H5 | Input image size mismatch (train vs export vs inference) | **Not present** — 640×640 is consistent across training, export, and the reference server inference path. |

## Contract self-check

An independent, from-scratch implementation of the full decode recipe in `TFLITE_CONTRACT.md` (de-normalize ×640 → convert center-format to corners → remove letterbox padding → per-class non-max suppression) was compared against the reference Ultralytics decoder on the same image:

- Detection counts: **12 vs 12** (at a low confidence threshold, to produce a non-empty comparison set)
- Class agreement: **all identical**
- Maximum confidence difference: **`4.96e-07`**
- Minimum per-box IoU: **`0.9998`**

The decode recipe in the contract is therefore verified to reproduce the reference decoder.

## Deliverables

- **`TFLITE_CONTRACT.md`** — the exact input/output specification and decode recipe for the device-side implementation.
- **`golden/`** — a golden sample bundle (input tensor, raw output, letterbox parameters, top pre-NMS candidates, and expected detections) so the device implementation can be validated stage-by-stage without sharing code.
- **`Names_Model_2026-06-27_14-44-47.txt`** — the authoritative 339-entry class list.
- **the `.tflite` model** — included so testing is done against the exact verified binary.

> The golden sample in this bundle uses a generic reference image for numerical validation of the decode path. It will be regenerated with a representative fish image once one is available; the recipe and validation method are unchanged.
