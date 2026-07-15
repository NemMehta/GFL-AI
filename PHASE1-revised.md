# Phase 1 (revised) — post-recon

**Supersedes Phases 1–3 of the original plan.** Phase 0 established that the TFLite consumer is the Android app and is not in this workspace. The original "PT vs TFLite pipeline diff" is therefore only half-executable, and the goal shifts:

> **Goal: determine whether the exported graph is faithful. If it is, produce an I/O contract spec that the Android dev can implement against.**

---

## Step 0 — Unblock (30 min, do this first)

- [ ] Scratch env: `pip install ultralytics tensorflow numpy opencv-python`. Use a fresh venv; do **not** try to reconstruct the pinned export env yet.
- [ ] Images: **you do not need a labeled val set for the control experiment.** 10–20 representative images (varied species, sizes, lighting) are enough, because the test compares `.pt` against `.tflite` — not against ground truth. Grab any fish photos available; if the client's val set can be recovered from Blob later, that's a bonus for the final mAP number, not a prerequisite.

---

## Step 1 — Interpreter I/O dump (needs `tensorflow` only, no images)

```python
import tensorflow as tf
i = tf.lite.Interpreter("Model_2026-06-27_14-44-47_float32.tflite"); i.allocate_tensors()
for d in i.get_input_details():  print("IN ", d["shape"], d["dtype"], d["quantization"])
for d in i.get_output_details(): print("OUT", d["shape"], d["dtype"], d["quantization"])
```

Record verbatim. This is the single most valuable artifact in the whole engagement — it resolves H1 (layout/decode-axis) on the export side and it is the backbone of the deliverable spec. Expect input `[1,640,640,3]` NHWC float32; the output will be either `[1,343,8400]` or `[1,8400,343]` and **which one it is** is the answer to the strongest hypothesis.

## Step 2 — Control experiment (the problem-splitter)

```python
from ultralytics import YOLO
pt, tfl = YOLO("...pt"), YOLO("..._float32.tflite")   # Ultralytics wraps tflite w/ correct pre+post
for img in images:
    a, b = pt(img, imgsz=640, conf=0.25)[0], tfl(img, imgsz=640, conf=0.25)[0]
    # compare: n detections, class ids, conf values, box IoU per matched pair
```

Report per image: detection count delta, class-ID agreement, mean IoU of matched boxes, mean conf delta.

**Fork:**
- **Outputs agree (IoU > ~0.95, same classes)** → the exported graph is faithful. H2 ruled out. The bug is entirely in the Android pre/post-processing. → go to Step 4.
- **Outputs disagree** → the export itself is lossy. H2 confirmed. → go to Step 3.

## Step 3 — Only if outputs disagree: re-export with matched versions

- [ ] Fresh env with `ultralytics==8.3.109` (match **training**, not the old export env) and a current `onnx2tf`.
- [ ] `YOLO("...pt").export(format="tflite", imgsz=640)` — pass `imgsz` explicitly rather than relying on the default.
- [ ] Re-run Step 2 against the new `.tflite`. If it now agrees → root cause is exporter/toolchain skew; the fix is a pinned, matched export env. Ship that as the finding.
- [ ] If it *still* disagrees, bisect the chain: export to ONNX, verify ONNX vs `.pt` with `onnxruntime`. ONNX matches but TFLite doesn't → fault is in `onnx2tf`. ONNX already differs → fault is upstream in the torch→ONNX step.

## Step 4 — Deliverable: the TFLite I/O contract

Regardless of which fork you land in, write `TFLITE_CONTRACT.md` and hand it to whoever owns the Android app:

- **Input:** exact shape, dtype, layout (NHWC), value range (`/255` → [0,1]), channel order (RGB), letterbox algorithm — resize preserving aspect to fit 640, pad with **114,114,114**, note where the padding offsets land.
- **Output:** exact shape and what each axis means. State plainly whether a transpose is required before decode.
- **Decode:** box format (xywh, center-based, in 640-space pixels), how to un-letterbox back to original image coords (subtract pad, divide by scale).
- **NMS:** `conf=0.25`, `iou=0.7`, `max_det=300`, class-agnostic **off** — matching the server defaults at `fish.py:484`, so device and server agree.
- **Classes:** 339 entries, order per `Names_Model_….txt` (== `data_338.yaml` `names:`). Explicitly warn against any hardcoded label list.
- Attach a **golden sample**: one image + its exact preprocessed input tensor + the raw output tensor + the expected final detections. This lets the Android dev bisect their own pipeline the same way, without you needing their code.

---

## Notes on scope

- H3 (class ordering) and H4 (imgsz) are ruled out on the export side. Keep them listed as **Android-side risks** in the deliverable — they are cheap for the Android dev to check and are classic causes of exactly this symptom.
- If the fork lands on "bug is in Android," say so to the client early and plainly. Fixing it requires access to that codebase; that's a scope change, not a failure. The contract spec in Step 4 is the value you deliver either way.
