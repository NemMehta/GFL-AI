# TFLITE_CONTRACT — build instructions + template

**For the agent.** Produce the final deliverable `TFLITE_CONTRACT.md` + a `golden/` bundle by:
1. Filling every `{{PLACEHOLDER}}` below with measured values from the workspace.
2. Generating the golden bundle per Appendix A.
3. Running the **mandatory self-check** (Appendix A, step 5). Do not ship if it fails — fix the contract math first.

The audience is the Android developer who owns the on-device decoder. They have the `.tflite` and this document; assume they have nothing else. Every claim must be implementable without reading our Python code.

---

## Contract template (fill and ship as TFLITE_CONTRACT.md)

### 1. Model identity

- File: `Model_2026-06-27_14-44-47_float32.tflite` — sha256 `{{TFLITE_SHA256}}`
- Class list: `Names_Model_2026-06-27_14-44-47.txt` — sha256 `{{NAMES_SHA256}}`, 339 entries, index order matches `data_338.yaml` `names:`. **Do not use any other label list.** (idx 0 = `{{NAME_0}}`, idx 338 = `{{NAME_338}}`)
- FP32, no quantization (`quantization=(0.0, 0)`). Never dequantize.
- Faithfulness verified against the source `.pt` by raw-tensor parity (class scores identical within float32 precision; boxes differ only by the deterministic /640 normalization documented below). See FINDINGS.md.

### 2. Input contract

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
- Note: your platform's bilinear resize will not be bit-identical to OpenCV's. That is acceptable; §7 shows how to test your decode path independently of preprocessing.

### 3. Output contract

- Tensor 0: shape `[1, 343, 8400]`, float32. **Verify the shape at runtime; do not assume.**
- Axis meaning: `343 = 4 box attributes + 339 class scores` (rows); `8400` = candidate predictions (columns).
  - Rows 0–3: box `cx, cy, w, h` — **center-format xywh, normalized to [0,1]** in 640×640 letterbox space. **Multiply by 640** (cx, w × input width; cy, h × input height) to get pixels. This is the convention the current decoder is suspected of missing.
  - Rows 4–342: per-class scores, **already sigmoid-activated**, each in `[0,1]`. There is **no objectness row**. Do not apply sigmoid or softmax again.
- Boxes are **fully decoded in-graph** (DFL + anchor/stride math already applied). Do not port YOLOv5-style anchor/grid decoding — there are no anchors to apply.
- Flat-buffer indexing: for a flat float array of length `343*8400`, element `(row r, prediction a)` is at offset `r * 8400 + a`. The classic bug is indexing `a * 343 + r`, i.e. assuming `[1, 8400, 343]`.
- Runtime sanity asserts (cheap, catch both known failure modes):

```
assert outputShape == [1, 343, 8400]
assert max(rows 0..3) <= 1.001          # if you see values >> 1, you are mis-indexing or mis-assuming pixels
assert 0 <= max(rows 4..342) <= 1.0     # if scores look uniform/garbage, axis order is wrong
```

### 4. Reference decode (language-neutral)

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

### 5. NMS parameters (must match server, `fish.py:484`)

- conf threshold **0.25** applied to the max class score
- IoU threshold **0.7**, **per-class** (class-agnostic OFF)
- max detections **300**

### 6. Numerics note

Run parity tests on **CPU (XNNPACK)**. GPU/NNAPI delegates may legally change float results beyond the tolerances in §7; validate correctness on CPU first, then enable delegates.

### 7. Golden sample — how to bisect your pipeline without sharing code

Bundle contents (`golden/`):

| File | What it is |
|---|---|
| `image.jpg` | source image ({{GOLDEN_IMAGE_NOTE}}) |
| `input_tensor.npy` | `[1,640,640,3]` float32 — the exact model input |
| `letterbox.json` | `r`, `pad_left`, `pad_top`, `W0`, `H0` for this image |
| `raw_output.npy` | `[1,343,8400]` float32 from the interpreter |
| `topk_anchors.csv` | top 20 predictions pre-NMS: `a, cx,cy,w,h (normalized), cls_idx, cls_name, score` |
| `final_detections.json` | boxes in **original-image pixels** after §4 decode + §5 NMS |

Three isolated tests, in order:

1. **Graph test** (bypasses your preprocessing): feed `input_tensor.npy` directly into your interpreter → compare against `raw_output.npy`. Pass: max abs diff ≤ 1e-3 on CPU. Fail here → interpreter/delegate issue, not your code.
2. **Decode test** (bypasses your preprocessing AND the model): run your decoder on `raw_output.npy` → compare against `topk_anchors.csv` and `final_detections.json`. Pass: same classes, score diff ≤ 0.02, per-box IoU ≥ 0.9. **This is the test expected to fail today** — if your boxes come out ~640× too small, the ×640 is missing; if classes/scores are garbage, check the stride in §3.
3. **Preprocessing test**: run your full pipeline on `image.jpg` → compare your input tensor against `input_tensor.npy` (mean abs diff ≤ 0.01; resize differences are expected) and your final detections against `final_detections.json` (IoU ≥ 0.9).

---

## Appendix A — golden bundle generation (agent instructions)

Write `make_golden.py` in the scratch venv:

1. Image: use a client fish photo if one has arrived; else `bus.jpg` with `{{GOLDEN_IMAGE_NOTE}}` = "content irrelevant; used for tensor parity — detections may be semantically meaningless".
2. Preprocess with Ultralytics' own `LetterBox` (import it; do not re-implement) → save `input_tensor.npy` + `letterbox.json`.
3. Run `tf.lite.Interpreter` on it → save `raw_output.npy`. Dump `topk_anchors.csv` (top 20 by max class score, pre-NMS, straight from the raw tensor).
4. Implement §4 exactly as written (independent code, not Ultralytics NMS) → `final_detections.json`.
5. **Mandatory self-check:** run `YOLO("...tflite").predict(image, imgsz=640, conf=0.25, iou=0.7, max_det=300)` and compare against your step-4 output. Require: identical detection count, identical classes, score diff ≤ 1e-3, per-box IoU ≥ 0.99. If this fails, the contract's decode math is wrong — fix §4 (and this appendix) before shipping anything.
6. Compute the two sha256 hashes, fill all `{{PLACEHOLDERS}}`, write final `TFLITE_CONTRACT.md` + `golden/`, and append a FINDINGS.md entry: "contract self-check passed" with the numbers.
