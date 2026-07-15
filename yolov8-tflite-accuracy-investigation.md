# YOLOv8 `.pt` → `.tflite` Accuracy Gap — Investigation Plan

**How to use:** drop this file at the repo root and open Claude Code with:
> `Read yolov8-tflite-accuracy-investigation.md and execute Phase 0 and Phase 1. Do not write fixes yet — report findings first.`

---

## Context for the agent

- A YOLOv8 model exported via Ultralytics `model.export(format='tflite')` (FP32, **not** quantized) shows lower detection accuracy than the source `.pt`.
- Quantization is ruled out as the cause. The gap is almost certainly in the **surrounding pipeline**, not the weights.
- Ranked suspects:
  1. **Preprocessing mismatch** — letterbox vs plain resize, padding color, `/255` normalization, BGR vs RGB, `imgsz` mismatch between export and inference.
  2. **Tensor layout** — TFLite exports are NHWC; the `.pt` path is NCHW. A missing/incorrect transpose silently produces garbage-but-plausible outputs.
  3. **Output decoding** — YOLOv8 head emits `[1, 84, 8400]` (or transposed). Wrong axis assumption → boxes decode to near-misses, tanking IoU-based mAP without obviously breaking.
  4. **Box format / scaling** — xywh vs xyxy, normalized [0,1] vs pixel coords, and failure to un-letterbox back to original image dims.
  5. **NMS + thresholds** — Ultralytics defaults (`conf=0.25`, `iou=0.7`, `max_det=300`, class-agnostic off) vs whatever the custom TFLite path uses.
  6. **Class ordering** — `data.yaml` order vs any hardcoded label list in the deployment code.

## Ground rules

- **Do not change model weights or re-train.** The `.pt` is the reference; the `.tflite` must match it.
- Report evidence before proposing fixes. Every claim gets a file:line citation or a printed tensor.
- Keep a running `FINDINGS.md` with: hypothesis → test → result → verdict (confirmed / ruled out).

---

## Phase 0 — Recon (read-only)

Produce a map of both inference paths, side by side.

- [ ] Locate: the export script/command, the `.pt` eval path, the `.tflite` eval path, `data.yaml`, and any deployment/runtime wrapper.
- [ ] For **each** path, extract and tabulate:
  - input size (`imgsz`), resize method, padding value, aspect-ratio handling
  - normalization (`/255`? mean/std? none?)
  - channel order (RGB/BGR) and where the conversion happens (`cv2.imread` gives BGR)
  - dtype and layout of the tensor actually handed to the model
  - output tensor shape as the code *assumes* it, vs. what it *is*
  - decode logic, box format, coordinate space
  - NMS implementation, `conf`, `iou`, `max_det`, class-agnostic flag
- [ ] Print the interpreter's real I/O spec:
  ```python
  import tensorflow as tf
  i = tf.lite.Interpreter(model_path="model.tflite"); i.allocate_tensors()
  print(i.get_input_details()); print(i.get_output_details())
  ```
- [ ] Check pinned versions: `ultralytics`, `torch`, `tensorflow` / `tflite-runtime`, `onnx`, `onnx2tf`. Version skew across the export toolchain is a real failure mode.

**Deliverable:** a two-column diff table (PT path | TFLite path) with every row where they disagree flagged.

---

## Phase 1 — Reproduce & establish the baseline

- [ ] Run both models on the labeled val set with the **existing** scripts. Record mAP50, mAP50-95, precision, recall. Confirm the gap is real and quantify it.
- [ ] **Control experiment:** load the `.tflite` back through the Ultralytics wrapper:
  ```python
  from ultralytics import YOLO
  YOLO("model.tflite").val(data="data.yaml", imgsz=<same as export>)
  ```
  - If accuracy **recovers** → the exported graph is fine; the bug is 100% in the custom pre/post-processing. Go straight to Phase 2.
  - If it's **still low** → the export itself is lossy (bad `imgsz`, NMS baked in wrong, opset/onnx2tf issue). Investigate the export command and toolchain versions instead.

This single test splits the problem in half. Do it before anything else.

---

## Phase 2 — Bisect the pipeline on one image

Pick one image where the `.pt` detects correctly and the `.tflite` doesn't. Then compare tensors at each stage:

- [ ] **Input tensor** — dump both preprocessed tensors. Compare `shape`, `dtype`, `min/max/mean`, and a few pixel values. `np.allclose` after transposing NHWC→NCHW. Any mismatch here is the bug; stop and fix.
- [ ] **Raw output tensor (pre-NMS)** — dump both. Compare shape, then max objectness/class score and its index. If the raw outputs match but final detections don't → the bug is in decode/NMS, not the model.
- [ ] **Post-decode boxes (pre-NMS)** — compare box coords in the same coordinate space. Look for tell-tale patterns: boxes off by the letterbox padding offset, boxes scaled by `imgsz/orig_size`, or x/y swapped.
- [ ] **Post-NMS** — compare with *identical* thresholds forced on both sides. If the gap only appears here, it's threshold/NMS config.

**Rule:** the first stage where the tensors diverge is the bug. Don't theorize past it.

---

## Phase 3 — Fix & verify

- [ ] Implement the minimal fix at the divergence point. Prefer reusing Ultralytics' own `LetterBox` and `non_max_suppression` over hand-rolled equivalents.
- [ ] Re-run the full val set. Target: `.tflite` mAP50-95 within ~1% of `.pt` (small residual from FP32 op-fusion differences is expected and acceptable).
- [ ] Add a regression guard: a test asserting pre-NMS tensor parity between the two paths on a fixed sample image, tolerance `atol=1e-3`.
- [ ] Document the root cause + fix in `FINDINGS.md`.

---

## Escape hatches (if Phase 2 shows no divergence)

- Per-class mAP breakdown — a single collapsed class points at class-ID ordering, not preprocessing.
- Confusion matrix — mass in the background column means detections are being dropped (threshold/NMS); mass off-diagonal means class remapping.
- Sweep `conf` from 0.001 upward on the TFLite path. If mAP climbs to match, detections existed but were filtered.
- Re-export with an explicit `imgsz=` matching inference, and try `format='onnx'` as an intermediate sanity check — if ONNX matches `.pt` but TFLite doesn't, the fault is in the ONNX→TF conversion step.
