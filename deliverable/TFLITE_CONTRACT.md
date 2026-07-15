# TFLite I/O Contract — GFL Fish Detector (YOLOv8n, 339 classes)

**Audience:** the Android developer who owns the on-device decoder. You have the `.tflite` and this document; assume nothing else. Every claim here is implementable without reading our Python code.

**Bottom line:** the `.pt` and `.tflite` were verified numerically equivalent — **class scores are identical within float32 precision (max abs diff 7.7e-07)**; the only difference is that **box coordinates come out normalized to `[0,1]` and must be multiplied by 640**. The on-device accuracy gap is a **decode bug, not a model/quantization problem**. Implement §2–§5 verbatim; use §7 to bisect your current pipeline.

> **THE LIKELY ONE-LINE FIX:** the 4 box values are `xywh ÷ 640`. **Multiply them by 640** before converting to corners. If your code treats them as pixels, every box is ~640× too small → detections are lost downstream → the accuracy collapse you're seeing.

---

## 1. Model identity

- Model: `Model_2026-06-27_14-44-47_float32.tflite` — sha256 `cc48c5972e592830a917a55cc4eeff33aee517e2474d6e94b0446ec0bf845997`
- Class list: `Names_Model_2026-06-27_14-44-47.txt` — sha256 `4ed8f711e6cb2891f20a3a0685ecea1ea3c75fb48865ccfb24918464aad9389e`, 339 entries, index order matches `data_338.yaml` `names:`. **Do not use any other label list.** (idx 0 = `black_fin_tuna_thunnus_atlanticus`, idx 338 = `irish_mojarra_diapterus_auratus`)
- FP32, no quantization (`quantization=(0.0, 0)`). Never dequantize.
- Faithfulness verified against the source `.pt` by raw-tensor parity: class scores identical within float32 precision (max abs diff `7.7e-07`); boxes differ only by the deterministic `/640` normalization documented below (median ratio exactly `640.0000`). See `VERIFICATION_SUMMARY.md`.

## 2. Input contract

- Tensor 0: shape `[1, 640, 640, 3]`, **NHWC**, float32, values in `[0, 1]`, channel order **RGB**. Bind by index, not name (name observed: `inputs_0`, but names vary by converter).
- Preprocessing = Ultralytics LetterBox, exactly:

```
r        = min(640 / W0, 640 / H0)          # W0,H0 = original image dims
newW     = round(W0 * r);  newH = round(H0 * r)
resize image → (newW, newH), bilinear interpolation
dw       = (640 - newW) / 2;  dh = (640 - newH) / 2
pad_left = round(dw - 0.1);  pad_right  = round(dw + 0.1)
pad_top  = round(dh - 0.1);  pad_bottom = round(dh + 0.1)
pad value = 114 per channel (= 0.4471 after /255)
then: RGB, float32, divide by 255.0
```

- No mean/std normalization. No BGR. `pad_left/pad_top/r` must be kept — they are needed to map boxes back (§4).
- Note: your platform's bilinear resize will not be numerically identical to OpenCV's. That is acceptable; §7 shows how to test your decode path independently of preprocessing.
- *(Worked value for the golden `image.jpg` = `bus.jpg`, 810×1080: `r = 0.592593`, `newW,newH = 480,640`, `pad_left = pad_right = 80`, `pad_top = pad_bottom = 0`.)*

## 3. Output contract

- Tensor 0: shape `[1, 343, 8400]`, float32. **Verify the shape at runtime; do not assume.**
- Axis meaning: `343 = 4 box attributes + 339 class scores` (rows); `8400` = candidate predictions (columns).
  - Rows 0–3: box `cx, cy, w, h` — **center-format xywh, normalized to [0,1]** in 640×640 letterbox space. **Multiply by 640** (cx, w × input width; cy, h × input height; both 640 here) to get pixels. This is the convention the current decoder is suspected of missing.
  - Rows 4–342: per-class scores, **already sigmoid-activated**, each in `[0,1]`. There is **no objectness row**. Do not apply sigmoid or softmax again.
- Boxes are **fully decoded in-graph** (DFL + anchor/stride math already applied). Do not port YOLOv5-style anchor/grid decoding — there are no anchors to apply.
- Flat-buffer indexing: for a flat float array of length `343*8400`, element `(row r, prediction a)` is at offset `r * 8400 + a`. The classic bug is indexing `a * 343 + r`, i.e. assuming `[1, 8400, 343]`.
- Runtime sanity asserts (cheap, catch both known failure modes):

```
assert outputShape == [1, 343, 8400]
assert max(rows 0..3) <= 1.001          # if you see values >> 1, you are mis-indexing or mis-assuming pixels
assert 0 <= max(rows 4..342) <= 1.0     # if scores look uniform/garbage, axis order is wrong
```

## 4. Reference decode (language-neutral)

```
# out: float32[343][8400] after dropping batch dim
CONF = 0.25
for a in 0..8399:
    c*  = argmax over c in 0..338 of out[4+c][a]
    s   = out[4+c*][a]
    if s < CONF: continue
    cx = out[0][a] * 640;  cy = out[1][a] * 640
    w  = out[2][a] * 640;  h  = out[3][a] * 640
    x1 = cx - w/2;  y1 = cy - h/2;  x2 = cx + w/2;  y2 = cy + h/2
    candidates += (x1, y1, x2, y2, s, c*)          # still in 640-space

per-class NMS(candidates, IoU = 0.7), keep top 300 by score      # in 640-space, matching reference

for each kept box:                                  # un-letterbox to original image coords
    x1 = (x1 - pad_left) / r;   x2 = (x2 - pad_left) / r
    y1 = (y1 - pad_top)  / r;   y2 = (y2 - pad_top)  / r
    clip x to [0, W0], y to [0, H0]
```

**Worked single anchor** (golden image, anchor 8329) — every intermediate value, so you can unit-test each step:
```
raw rows 0..3 (normalized) : [0.501803, 0.818859, 0.749499, 0.362634]
best class                 : 86 (duckbill_ray_aetomylaeus_bovinus)  score 0.080881
× 640  (cx,cy,w,h)         : 321.154, 524.070, 479.679, 232.086
→ xyxy @640 letterbox      : 81.314, 408.027, 560.993, 640.113
→ un-letterbox to original : 2.22, 688.55, 810.00, 1080.00
```

## 5. NMS parameters (must match server, `gfl-python/app_backend/routes/fish.py:484`)

- conf threshold **0.25** applied to the max class score
- IoU threshold **0.7**, **per-class** (class-agnostic OFF)
- max detections **300**

## 6. Numerics note

Run parity tests on **CPU (XNNPACK)**. GPU/NNAPI delegates may legally change float results beyond the tolerances in §7; validate correctness on CPU first, then enable delegates.

## 7. Golden sample — bisect your pipeline without sharing code

Bundle contents (`golden/`):

| File | What it is |
|---|---|
| `image.jpg` | source image (content irrelevant — `bus.jpg` stand-in used for tensor parity; **has no fish**, so detections are semantically meaningless but numerically authoritative). Regenerate with a real fish image when one is available. |
| `input_tensor.npy` | `[1,640,640,3]` float32 — the exact model input |
| `letterbox.json` | `r`, `pad_left`, `pad_top` (+ `pad_right/bottom`), `W0`, `H0` for this image |
| `raw_output.npy` | `[1,343,8400]` float32 from the interpreter |
| `topk_anchors.csv` | top 20 predictions pre-NMS: `a, cx,cy,w,h (normalized), cls_idx, cls_name, score` |
| `final_detections.json` | boxes in **original-image pixels** after §4 decode + §5 NMS at conf=0.25 (**empty** for `bus.jpg` — no fish) |
| `final_detections_conf0.001.json` | same at conf=0.001 (12 boxes) — a **non-empty** set for the decode test below |

Three isolated tests, in order:

1. **Graph test** (bypasses your preprocessing): feed `input_tensor.npy` directly into your interpreter → compare against `raw_output.npy`. Pass: max abs diff ≤ 1e-3 on CPU. Fail here → interpreter/delegate issue, not your code.
2. **Decode test** (bypasses your preprocessing AND the model): run your decoder on `raw_output.npy` → compare against `topk_anchors.csv` (pre-NMS) and `final_detections_conf0.001.json` (post-NMS, non-empty). Pass: same classes, score diff ≤ 0.02, per-box IoU ≥ 0.9. **This is the test expected to fail today** — if your boxes come out ~640× too small, the ×640 is missing; if classes/scores are garbage, check row indexing in §3.
3. **Preprocessing test**: run your full pipeline on `image.jpg` → compare your input tensor against `input_tensor.npy` (mean abs diff ≤ 0.01; resize differences are expected) and your final detections against `final_detections_conf0.001.json` (IoU ≥ 0.9). Note: at conf=0.001 the golden detections are borderline, so a ±1–2 difference in detection count from platform resize differences is expected and is not a failure. Test 2, which bypasses preprocessing, is the deterministic check. This tolerance tightens once the bundle is regenerated with a real fish image.

**Contract self-check (already run on our side):** an independent from-scratch implementation of §4–§5 was compared against `YOLO("...tflite").predict(imgsz=640, iou=0.7, max_det=300)`. Result: at conf=0.001, **12 vs 12 detections, identical classes, max score diff `4.96e-07`, min per-box IoU `0.9998`** (at conf=0.25 both produce 0 — `bus.jpg` has no fish). The decode math above is therefore verified correct and complete.

---

## Android-side risks worth a 5-minute check (cheap, classic causes of this exact symptom)

- **Class list drift:** if the app bundles its own hardcoded label array and it diverged from `Names_Model_….txt`, every ID maps to the wrong species. Load labels from the shipped names file that pairs with the model.
- **Layout assumption:** confirm you read `[1,343,8400]` (attributes-first), not `[1,8400,343]`.
- **Input size:** confirm the app feeds 640×640 (letterboxed), not a plain resize or a different size.
