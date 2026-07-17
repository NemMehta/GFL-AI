#!/usr/bin/env python3
"""Verify a GFL fish-detector .tflite against its golden bundle.

Self-contained on purpose: needs only `tensorflow` (or `tflite-runtime`) and
`numpy`. No torch, no ultralytics, no repo checkout. Point it at the bundle and
it prints PASS/FAIL per stage.

    python verify_tflite.py --model <model.tflite> --golden <golden/>

Stages, in the order you should trust them:

  1. contract    -- input/output shape + dtype are what the contract claims
  2. convention  -- are box rows PIXELS or NORMALIZED? auto-detected and reported
  3. graph       -- feed golden/input_tensor.npy, compare to golden/raw_output.npy.
                    Bypasses your preprocessing. Fails here => interpreter or
                    delegate problem, not your code.
  4. decode      -- decode golden/raw_output.npy, compare to
                    golden/final_detections.json. Bypasses preprocessing AND the
                    model. This is the stage that isolates decode bugs.

Exit code 0 = all passed, 1 = something failed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf

    _Interpreter = tf.lite.Interpreter
except ImportError:  # tflite-runtime is the lighter option
    from tflite_runtime.interpreter import Interpreter as _Interpreter

CONF = 0.25
IOU = 0.7
MAX_DET = 300


# ---------------------------------------------------------------- decode ----
def iou_vec(box, boxes):
    """IoU of one xyxy box against an (N,4) array."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms_per_class(boxes, scores, classes, iou_thr=IOU, max_det=MAX_DET):
    """Greedy per-class NMS. Equivalent to torchvision.ops.batched_nms."""
    keep = []
    order = np.argsort(-scores)
    for c in np.unique(classes):
        idx = order[classes[order] == c]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            idx = idx[1:][iou_vec(boxes[i], boxes[idx[1:]]) <= iou_thr]
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(-scores[keep])][:max_det]


def decode(raw, r, pad_left, pad_top, scale, conf_th=CONF):
    """[1,343,8400] -> [(cls, score, [x1,y1,x2,y2] in ORIGINAL image px)].

    `scale` is 1.0 for a pixel-box model and 640 for a normalized one.
    """
    p = raw[0]
    b = p[:4].astype(np.float64) * scale
    s = p[4:]
    cls, conf = s.argmax(0), s.max(0)
    x1, y1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    x2, y2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    k = conf > conf_th
    if k.sum() == 0:
        return []
    boxes = np.stack([x1, y1, x2, y2], 1)[k]
    scores, classes = conf[k], cls[k]
    out = []
    for i in nms_per_class(boxes, scores, classes):
        x1_, y1_, x2_, y2_ = boxes[i]
        out.append(
            (
                int(classes[i]),
                round(float(scores[i]), 4),
                [
                    round(float((x1_ - pad_left) / r), 1),
                    round(float((y1_ - pad_top) / r), 1),
                    round(float((x2_ - pad_left) / r), 1),
                    round(float((y2_ - pad_top) / r), 1),
                ],
            )
        )
    return sorted(out, key=lambda t: -t[1])


def detect_convention(raw):
    """Infer the box convention from the box rows themselves.

    The two conventions sit ~640x apart, so this is unambiguous. Note a
    normalized row can legitimately exceed 1.0 (a box may extend past the
    letterbox edge; measured max 1.0017) -- do NOT assert <= 1.0 exactly.
    """
    m = float(np.abs(raw[0, :4]).max())
    return ("pixel" if m > 10.0 else "normalized"), m


# ----------------------------------------------------------------- stages ----
def load_golden(g: Path):
    lb = json.loads((g / "letterbox.json").read_text())
    fd = json.loads((g / "final_detections.json").read_text())
    return (
        np.load(g / "input_tensor.npy"),
        np.load(g / "raw_output.npy"),
        lb,
        fd,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--names", type=Path, help="class-name list, one per line")
    ap.add_argument("--graph-tol", type=float, default=1e-3)
    args = ap.parse_args()

    for p in (args.model, args.golden):
        if not p.exists():
            raise SystemExit(f"no such path: {p}")

    x, raw_ref, lb, dets_ref = load_golden(args.golden)
    results = {}

    it = _Interpreter(model_path=str(args.model))
    it.allocate_tensors()
    IN, OUT = it.get_input_details()[0], it.get_output_details()[0]

    # 1. contract
    in_ok = list(IN["shape"]) == [1, 640, 640, 3] and IN["dtype"] == np.float32
    out_ok = list(OUT["shape"]) == [1, 343, 8400] and OUT["dtype"] == np.float32
    print(f"  input  {list(IN['shape'])} {IN['dtype'].__name__} quant={IN['quantization']}")
    print(f"  output {list(OUT['shape'])} {OUT['dtype'].__name__} quant={OUT['quantization']}")
    results["contract"] = in_ok and out_ok

    # 2. convention
    it.set_tensor(IN["index"], x.astype(IN["dtype"]))
    it.invoke()
    raw = it.get_tensor(OUT["index"])
    conv, mx = detect_convention(raw)
    scale = 1.0 if conv == "pixel" else 640.0
    conv_ref, _ = detect_convention(raw_ref)
    print(f"  convention: {conv.upper()} (box-rows max {mx:.4f}) -> decode scale x{scale:g}")
    if conv != conv_ref:
        print(f"  !! golden raw_output.npy is {conv_ref.upper()} -- bundle mismatch")
    results["convention"] = conv == conv_ref

    # 3. graph
    d = float(np.abs(raw - raw_ref).max())
    print(f"  graph: max abs diff vs golden raw_output.npy = {d:.3e} (tol {args.graph_tol:g})")
    results["graph"] = d <= args.graph_tol

    # 4. decode -- a pure test of decode logic, independent of the model. Uses the
    # golden's own convention, not the model's, so a convention mismatch shows up
    # as "stage 2/3 fail, stage 4 pass" = your decoder is fine, your model is wrong.
    conf_th = dets_ref.get("conf", CONF)
    golden_scale = dets_ref.get("decode_scale", 1.0 if conv_ref == "pixel" else 640.0)
    dets = decode(raw_ref, lb["r"], lb["pad_left"], lb["pad_top"], golden_scale, conf_th=conf_th)
    ref = [(d_["cls"], d_["conf"], d_["xyxy"]) for d_ in dets_ref["detections"]]
    count_ok = len(dets) == len(ref)
    cls_ok = [a[0] for a in dets] == [b[0] for b in ref]
    if dets and ref and count_ok:
        min_iou = min(
            float(iou_vec(np.array(a[2]), np.array([b[2]]))[0]) for a, b in zip(dets, ref)
        )
        max_sd = max(abs(a[1] - b[1]) for a, b in zip(dets, ref))
    else:
        min_iou, max_sd = (None, None)
    print(f"  decode: {len(dets)} dets vs {len(ref)} expected | classes_match={cls_ok} "
          f"| min IoU={min_iou} | max score diff={max_sd}")
    results["decode"] = count_ok and cls_ok and (min_iou is None or min_iou >= 0.9) and (
        max_sd is None or max_sd <= 0.02
    )

    if args.names and args.names.exists():
        names = args.names.read_text(encoding="utf-8").splitlines()
        print(f"  labels: {len(names)} entries; showing top detections")
        for c, s, _ in dets[:5]:
            print(f"     {s:.3f}  [{c}] {names[c] if c < len(names) else '??'}")

    print()
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(results.values())
    print(f"\n{'ALL STAGES PASSED' if ok else 'FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
