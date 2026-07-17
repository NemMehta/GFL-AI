# TFLite I/O Contract v2 — GFL Fish Detector (YOLOv8n, 339 classes)

**Audience:** the Android developer who owns the on-device decoder. You have the `.tflite`,
this document, and a `golden/` bundle. Assume nothing else.

> ## ⚠️ READ THIS IF YOU IMPLEMENTED v1
>
> **v1 told you to multiply the box outputs by 640. This model does NOT need that.**
> Its boxes already come out in **pixels**. If you apply the v1 `×640` to this model, every
> box will be **640× too large**. Delete the multiply.
>
> The two models are trivially distinguishable at runtime — see §3.1. If you would rather keep
> your `×640` code, keep using the **v1 normalized model** instead; both are correct, they just
> disagree about units. **Do not mix a v1 decoder with a v2 model.**

**What changed and why.** Ultralytics' TFLite exporter deliberately renormalizes box
coordinates to `[0,1]` (`head.py:65-72`, citing numerical stability for INT8 quantization).
That is legitimate but undocumented in the artifact, and a decoder that misses the `×640`
collapses every box — the on-device accuracy gap you reported. This build removes that
renormalization at export time, so the `.tflite` emits the **same pixel-space boxes the `.pt`
emits**. There is now no convention delta between server and device.

---

## 1. Model identity

- Model: `Model_2026-06-27_14-44-47_float32_pixelbox.tflite`
  sha256 `10ad8fccc977ad90fbc7b32d839302583d7d5c32e8da5c3602c426d0eb9df84b` (13,815,352 bytes)
- Source: `Model_2026-06-27_14-44-47.pt`, sha256 `59dcac6daae91da93310214d7d7dc906151b0f53fc7d7de108937271f3f30cc4`
- Labels: `labels.txt`, sha256 `18b159d0eefeb75739930fd9efb05b4d372e8b60344ab6aac94da667a61d856e`
  — **339 lines, plain, line *i* == class *i*.** Read §6 before touching any other label file.
- FP32, no quantization (`quantization=(0.0, 0)`). Never dequantize.
- Verified against the `.pt` on all 7 test images: box median ratio **`1.0000`**, class scores
  identical within float32 (max abs diff `6.6e-06`), detections identical, **per-box IoU `1.0`**.

### 1.1 Known delta vs the v1 model — no embedded metadata

The v1 model carried a metadata zip (`imgsz`, class names). Attaching it requires
`tflite-support`, which publishes **no Windows wheel**, and this build was produced on Windows.
So this artifact has **no embedded metadata**. Practical effect: your tooling cannot auto-populate
labels from the model — use the shipped `labels.txt`. Everything about the tensors is unchanged.
(To restore it, re-export on Linux with `pip install tflite-support==0.4.4`.)

## 2. Input contract

Identical to v1 — nothing here changed.

- Tensor 0: shape `[1, 640, 640, 3]`, **NHWC**, float32, values in `[0, 1]`, channel order **RGB**.
  Bind by index, not name. **No signature defs exist** (`serving_default` is absent) — onnx2tf
  models expose none.
- Preprocessing = Ultralytics LetterBox, exactly:

```
r        = min(640 / W0, 640 / H0)          # W0,H0 = original image dims
newW     = round(W0 * r);  newH = round(H0 * r)
resize image → (newW, newH), bilinear
dw       = (640 - newW) / 2;  dh = (640 - newH) / 2
pad_left = round(dw - 0.1);  pad_right  = round(dw + 0.1)
pad_top  = round(dh - 0.1);  pad_bottom = round(dh + 0.1)
pad value = 114 per channel (= 0.4471 after /255)
then: RGB, float32, divide by 255.0
```

- No mean/std normalization. No BGR. Keep `pad_left/pad_top/r` — needed to map boxes back (§4).
- Your platform's bilinear resize will not be bit-identical to OpenCV's. That's fine; §7 test 2
  checks your decode independently of preprocessing.

## 3. Output contract

- Tensor 0: shape `[1, 343, 8400]`, float32. **Verify at runtime; do not assume.**
- `343 = 4 box attributes + 339 class scores` (rows); `8400` = candidate predictions (columns).
  - Rows 0–3: box `cx, cy, w, h` — **center-format xywh, in PIXELS** in 640×640 letterbox space.
    **Use as-is. Do not scale.**
  - Rows 4–342: per-class scores, **already sigmoid-activated**, each in `[0,1]`. There is **no
    objectness row**. Do not apply sigmoid or softmax again.
- Boxes are **fully decoded in-graph** (DFL + anchor/stride math applied). Do not port
  YOLOv5-style anchor/grid decoding — there are no anchors to apply.
- Flat-buffer indexing: for a flat float array of length `343*8400`, element `(row r, prediction a)`
  is at `r * 8400 + a`. The classic bug is `a * 343 + r`, i.e. assuming `[1, 8400, 343]`.

### 3.1 Runtime sanity asserts

```
assert outputShape == [1, 343, 8400]
assert max(rows 0..3) > 100          # PIXEL model (this one): values run up to ~640
                                     # if instead max <= ~1.01, you have the v1 NORMALIZED
                                     # model and need the x640. Do not guess -- check.
assert 0 <= max(rows 4..342) <= 1.0  # if scores look uniform/garbage, axis order is wrong
```

> **Correction to v1.** v1 §3 told you to `assert max(rows 0..3) <= 1.001`. That assert is
> **wrong even for the v1 model** — a box may extend past the letterbox edge, so a normalized
> row legitimately exceeds 1.0 (measured: **1.0037**). It would have fired a false failure on
> real images. The two conventions sit ~640× apart; any threshold between ~10 and ~100
> separates them safely. Do not hair-split near 1.0.

## 4. Reference decode (language-neutral)

```
# out: float32[343][8400] after dropping batch dim
CONF = 0.25
for a in 0..8399:
    c*  = argmax over c in 0..338 of out[4+c][a]
    s   = out[4+c*][a]
    if s < CONF: continue
    cx = out[0][a];  cy = out[1][a]          # already pixels -- NO x640
    w  = out[2][a];  h  = out[3][a]
    x1 = cx - w/2;  y1 = cy - h/2;  x2 = cx + w/2;  y2 = cy + h/2
    candidates += (x1, y1, x2, y2, s, c*)    # still in 640-space

per-class NMS(candidates, IoU = 0.7), keep top 300 by score

for each kept box:                            # un-letterbox to original image coords
    x1 = (x1 - pad_left) / r;   x2 = (x2 - pad_left) / r
    y1 = (y1 - pad_top)  / r;   y2 = (y2 - pad_top)  / r
    clip x to [0, W0], y to [0, H0]
```

**Worked single anchor** (golden image, anchor 7072) — every intermediate value, so you can
unit-test each step:

```
raw rows 0..3 (pixels)     : [525.768066, 261.366394, 128.160736, 40.246002]
best class                 : 17 (Mangrove Snapper (Lutjanus griseus))  score 0.879173
                             (no x640 step -- this is where v1 had one)
→ xyxy @640 letterbox      : 461.688, 241.243, 589.848, 281.489
→ un-letterbox (r=1.28,
   pad_left=0, pad_top=127): 360.7, 89.3, 460.8, 120.7
```

## 5. NMS parameters (must match server, `gfl-python/app_backend/routes/fish.py:484`)

- conf threshold **0.25** applied to the max class score
- IoU threshold **0.7**, **per-class** (class-agnostic OFF)
- max detections **300**

## 6. Class labels — the off-by-one trap

**Use the shipped `labels.txt`: 339 lines, one name per line, line *i* is class *i*.**

The file v1 shipped, `Names_Model_2026-06-27_14-44-47.txt`, is **not** a plain label list — it is
**340 lines**: a `class_map:` header followed by `  <idx>: <Common Name> (<Binomial>)`:

```
class_map:
  0: Black fin Tuna (Thunnus atlanticus)
  1: Largemouth Bass (Micropterus nigricans)
```

The obvious `readLines()` therefore yields index 0 = `"class_map:"` and **shifts every species by
one** — class 17 reports as *Little Tunny* when it is actually *Mangrove Snapper*. v1 §1 called it
"339 entries", which invites exactly this bug. `labels.txt` is that file parsed and flattened; its
339 names were **cross-checked entry-by-entry against the model's own embedded metadata and all
339 agree**, so the ordering is authoritative.

If you keep your own hardcoded label array, it will drift. Load `labels.txt`.

## 7. Golden sample — bisect your pipeline without sharing code

Bundle (`golden/`) — generated from a **real fish image** with 10 confident detections at the real
`conf=0.25` threshold. (v1's golden was `bus.jpg`, which contains no fish: its
`final_detections.json` was empty and only a `conf=0.001` noise file was usable.)

| File | What it is |
|---|---|
| `image.webp` | source image, 500×301 |
| `input_tensor.npy` | `[1,640,640,3]` float32 — the exact model input |
| `letterbox.json` | `r=1.28`, `pad_left=0`, `pad_top=127`, `W0=500`, `H0=301` |
| `raw_output.npy` | `[1,343,8400]` float32 from the interpreter |
| `topk_anchors.csv` | top 20 pre-NMS: `a, cx,cy,w,h (PIXELS), cls_idx, cls_name, score` |
| `final_detections.json` | **10 boxes** in original-image pixels after §4 decode + §5 NMS at conf=0.25 |

Three isolated tests, in order:

1. **Graph test** (bypasses your preprocessing): feed `input_tensor.npy` into your interpreter →
   compare against `raw_output.npy`. Pass: max abs diff ≤ `1e-3` on CPU. Fail → interpreter or
   delegate issue, not your code.
2. **Decode test** (bypasses preprocessing AND the model): run your decoder on `raw_output.npy` →
   compare against `topk_anchors.csv` (pre-NMS) and `final_detections.json` (post-NMS). Pass: same
   classes, score diff ≤ 0.02, per-box IoU ≥ 0.9. **This is the deterministic check.**
3. **Preprocessing test**: run your full pipeline on `image.webp` → compare your input tensor
   against `input_tensor.npy` (mean abs diff ≤ 0.01; resize differences expected) and your final
   detections against `final_detections.json` (IoU ≥ 0.9).

### Run the reference verifier first

`verify_tflite.py` runs tests 1–2 for you. Needs only `numpy` + `tensorflow` (or `tflite-runtime`):

```
python verify_tflite.py --model Model_2026-06-27_14-44-47_float32_pixelbox.tflite \
                        --golden golden/ --names labels.txt
```

It auto-detects the box convention and prints PASS/FAIL per stage; exit code 0 = all passed. If it
passes for us and fails for you, the difference is your environment, not the model.

## 8. Numerics

Run parity tests on **CPU (XNNPACK)**. GPU/NNAPI delegates may legally change float results beyond
the §7 tolerances. Validate on CPU first, then enable delegates.

---

## Provenance

- Exported by `export/export_pixelbox.py`, which monkey-patches `Detect.forward` to drop the
  `head.py:65-72` renormalization. Toolchain pinned in `export/requirements-export.txt`:
  ultralytics 8.1.0, torch 2.1.2+cpu, tensorflow-cpu 2.13.1, onnx2tf 1.17.5.
- **This is a verification build.** The production Azure converter
  (`gfl-tflite-conversion/converter.py:51`) is unchanged and still emits **normalized** models.
  Any model re-exported through that service will be v1-convention. Check §3.1 rather than
  assuming which one you have.
