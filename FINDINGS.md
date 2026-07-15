# FINDINGS — YOLOv8 `.pt` → `.tflite` Accuracy Gap

Running log. Format: **hypothesis → test → result → verdict** (confirmed / ruled out / untestable-here / pending).

Investigation scope so far: **Phase 0 (recon, read-only) complete. Phase 1 blocked on missing assets** (no val set / test images on disk, no built env). See "Blockers" at the bottom.

---

## TOP-LINE FINDING (changes the whole approach)

**The "custom TFLite pre/post-processing path" that the plan's ranked suspects (letterbox, `/255`, NHWC transpose, decode, NMS, class order) are about does NOT exist anywhere in this Python workspace.**

- TFLite is only ever *produced* here — `gfl-tflite-conversion/converter.py:51` (`model.export(format="tflite")`). It is then uploaded to Azure Blob (`tflite/<name>_float32.tflite`) and **consumed on-device by the Android app**, which is not in this repo set.
- The only TensorFlow/tflite code in Python is the Keras *embedding* model in `services/uniqueness.py` — unrelated to detection.
- Server-side detection (`gfl-python`) runs the **`.pt`** through Ultralytics, never the `.tflite`.

**Consequence:** Phase 0's "two-column PT vs TFLite diff table" can only be half-filled from this workspace. The PT column is real code; the TFLite consumer column is Android code we can't see. The accuracy gap the user reports is almost certainly in that Android decode/NMS/preprocess layer OR in the export toolchain — and we can split those two with the Phase 1 control experiment (see below), which is still runnable here.

---

## Inference-path map (Phase 0 deliverable)

| Dimension | PT path (`gfl-python`, server) | TFLite path (Android, **not in repo**) |
|---|---|---|
| Entry point | `app_backend/routes/fish.py:470` `/detect_fish`; also `services/detect.py` | on-device, unavailable |
| Framework | Ultralytics `YOLO.predict` | raw `tf.lite.Interpreter` (assumed) |
| Input size | `imgsz=640` (`fish.py:484`) | unknown — must be 640 to match |
| Resize / aspect | Ultralytics `LetterBox` (auto) | unknown |
| Normalization | Ultralytics internal `/255` | unknown |
| Channel order | `PIL … .convert("RGB")` (`fish.py:481`); `detect.py` takes `img_bgr` (cv2/BGR — Ultralytics auto-handles) | unknown |
| Layout | NCHW (torch) | NHWC (tflite) — **transpose required** |
| Output shape | `[1, 343, 8400]` (4 box + 339 cls) | depends on `onnx2tf` — often `[1, 8400, 343]` |
| Decode / NMS | Ultralytics built-in | hand-rolled on-device |
| conf / iou | `conf=0.25`, iou default `0.7` (`fish.py:484`) | unknown |

---

## Export toolchain (Phase 0 deliverable)

`.pt` → ONNX → `onnx2tf` → TF SavedModel → `.tflite`, driven by `converter.py:51` `model.export(format="tflite")` — **no `imgsz` passed → defaults to 640.**

### Version pins across the toolchain — **SKEW PRESENT**

| Component | ultralytics | torch | tensorflow | onnx2tf | numpy |
|---|---|---|---|---|---|
| **Training** (`gfl-ml-training`) | **8.3.109** | 2.3.1 (commented) | 2.19.0 | — | 1.26.4 |
| **TFLite export** (`gfl-tflite-conversion`) | **8.1.0** | 2.1.2+cpu | 2.13.1 | **1.17.5** | 1.24.3 |
| Server `.pt` inference (`gfl-python`) | >=8.0.0 (unpinned) | — | >=2.13.0 | — | >=1.22.0 |
| CoreML export (`gfl-model-conversion`) | >=8.0.0 (unpinned) | — | — | — | — |

Model is **trained on Ultralytics 8.3.109 but exported by 8.1.0** — a real, documented failure mode. `onnx2tf==1.17.5` is old; onnx2tf version directly determines the TFLite output tensor **layout** (suspect #3).

---

## Hypotheses

### H1 — Output-layout / decode-axis mismatch (`onnx2tf` transpose)
- **Hypothesis:** `onnx2tf` emits the YOLOv8 head as `[1, 8400, 343]` (channels-last) while the on-device decoder assumes `[1, 343, 8400]` (or vice-versa). Wrong-axis decode → boxes land as near-misses → mAP collapses without an obvious crash. Amplified by the 8.1.0-vs-8.3.109 exporter/trainer skew.
- **Test:** dump the interpreter I/O spec (needs env); compare against Android decoder assumptions (needs Android code).
- **Result:** pending. **Verdict: pending (strongest candidate).**

### H2 — Export toolchain version skew
- **Hypothesis:** exporting an 8.3.109-trained model with the 8.1.0 exporter + old onnx2tf produces a subtly wrong graph.
- **Test:** Phase 1 control experiment (below).
- **Result:** pending. **Verdict: pending.**

### H3 — Class-ID ordering rot
- **Hypothesis:** `data_338.yaml` has `nc: 339` (filename says 338) and carries the known test/QA + duplicate-binomial rot. If Android uses a stale hardcoded label list, class IDs remap → per-class collapse.
- **Test / result:** The model ships a sidecar `Names_Model_….txt` with **339 entries in the same order as the yaml `names:`** (verified: idx 0 = Black fin Tuna, 1 = Largemouth Bass, … matches yaml). So the label list *shipped with the model* is self-consistent. Whether Android actually uses this file vs a stale one is **untestable here**.
- **Verdict: ruled out as an export-side bug; open as an Android-side risk.**

### H4 — imgsz mismatch
- **Hypothesis:** export imgsz ≠ inference imgsz.
- **Test/result:** Training `IMGSZ=640` (`config.py:398`, `train_models.py:572`); TFLite export defaults to 640; CoreML export `imgsz=640` (`convertion.py:242`); server inference `imgsz=640` (`fish.py:484`). All consistent at 640. Android input size unknown.
- **Verdict: ruled out on the export/server side; unverifiable for Android.**

---

## Assets on disk (usable for Phase 1)

- `Files_from_email/Model_2026-06-27_14-44-47.pt` — source `.pt` (7.0 MB)
- `Files_from_email/Model_2026-06-27_14-44-47_float32.tflite` — matched export (13.9 MB)
- `Files_from_email/data_338.yaml` — `nc: 339`, 339 names
- `Files_from_email/Names_Model_2026-06-27_14-44-47.txt` — 339-entry class_map, order matches yaml
- `Files_from_email/3dc87dbb-….mlpackage.zip` — CoreML export

**A matched `.pt` + `.tflite` pair is present** → the Phase 1 control experiment is runnable *if* we build the env and get a few images.

---

## Blockers (why Phase 1 is not yet run)

1. **No labeled val set and no test images anywhere on disk** → cannot compute the mAP gap (Phase 1, bullet 1) or run `YOLO(...).val(data=…)` locally. The val data lives in Azure Blob (SAS tokens in source are expired).
2. **No built Python env** (venvs were deleted). The interpreter I/O dump needs `tensorflow`; the control experiment needs `ultralytics`+`torch`.
3. **Android app code absent** → the actual custom TFLite decode/NMS/preprocess cannot be inspected.

### What IS runnable here once a scratch env + a few images exist
- **Interpreter I/O dump** (`tf.lite.Interpreter` → input/output details): reveals real TFLite layout/shape/dtype. (needs `tensorflow` only)
- **Control experiment (the problem-splitter):** run the same handful of images through `YOLO(pt)` and `YOLO(tflite)` (Ultralytics wraps the tflite with a *correct* pre/post). If the two agree → export graph is faithful → bug is 100% in Android. If they disagree → export/toolchain is lossy (H2). This does not need the full val set — a few images suffice to split the problem.
