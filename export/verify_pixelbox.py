"""Dev-side check that the pixelbox export is faithful to the .pt.

Answers three questions, in order:
  1. Did the convention actually flip?  box-row ratio pt:tflite should be 1.0
     for the pixelbox model and 640.0 for the original normalized one.
  2. Did anything else move?  class rows must stay identical within float32.
  3. Do real detections agree?  decode both at scale=1.0 over every test image;
     same classes, same count, per-box IoU >= 0.99.

Needs torch + ultralytics + tensorflow. The team-facing verifier
(deliverable-v2/verify_tflite.py) needs only tensorflow + numpy.

Usage:
  python export/verify_pixelbox.py --pt <model.pt> --pixelbox <new.tflite> \
      [--normalized <old.tflite>] --images <dir> [--out <report.json>]
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import tensorflow as tf

IMGSZ = 640


def load_image(p):
    im = cv2.imread(str(p))
    if im is None:  # cv2 can miss some .webp builds
        from PIL import Image

        im = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im


def letterbox_params(W0, H0):
    r = min(IMGSZ / W0, IMGSZ / H0)
    nW, nH = round(W0 * r), round(H0 * r)
    return r, round((IMGSZ - nW) / 2 - 0.1), round((IMGSZ - nH) / 2 - 0.1)


def preprocess(im0):
    x = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)(image=im0)[:, :, ::-1]
    return np.ascontiguousarray(x, np.float32) / 255.0


def decode(raw, r, pl, pt_, scale, conf_th=0.25):
    """xywh->xyxy, un-letterbox, per-class NMS. Mirrors evidence/same_tensor_decode.py."""
    p = raw[0]
    b = p[:4].astype(np.float64) * scale
    s = p[4:]
    cls, cf = s.argmax(0), s.max(0)
    x1, y1, x2, y2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    k = cf > conf_th
    if k.sum() == 0:
        return []
    B, C, K = np.stack([x1, y1, x2, y2], 1)[k], cf[k], cls[k]
    idx = torchvision.ops.batched_nms(
        torch.tensor(B, dtype=torch.float32),
        torch.tensor(C, dtype=torch.float32),
        torch.tensor(K),
        0.7,
    ).numpy()[:300]
    return sorted(
        [
            (
                int(K[i]),
                round(float(C[i]), 4),
                [
                    round(float((B[i, 0] - pl) / r), 1),
                    round(float((B[i, 1] - pt_) / r), 1),
                    round(float((B[i, 2] - pl) / r), 1),
                    round(float((B[i, 3] - pt_) / r), 1),
                ],
            )
            for i in idx
        ],
        key=lambda t: -t[1],
    )


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def interp(path):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


def run_tflite(it, IN, OUT, x):
    it.set_tensor(IN["index"], x[None].astype(IN["dtype"]))
    it.invoke()
    return it.get_tensor(OUT["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, type=Path)
    ap.add_argument("--pixelbox", required=True, type=Path)
    ap.add_argument("--normalized", type=Path, help="original model, as a control")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    m = YOLO(str(args.pt))
    m.model.eval()

    px_it, px_IN, px_OUT = interp(args.pixelbox)
    print(f"pixelbox   in={list(px_IN['shape'])} out={list(px_OUT['shape'])} dtype={px_OUT['dtype'].__name__}")
    nm = None
    if args.normalized:
        nm = interp(args.normalized)
        print(f"normalized in={list(nm[1]['shape'])} out={list(nm[2]['shape'])}")

    imgs = sorted(
        sum([glob.glob(str(args.images / e)) for e in ("*.webp", "*.jpg", "*.jpeg", "*.png")], [])
    )
    if not imgs:
        raise SystemExit(f"no images under {args.images}")

    rows = []
    print(f"\n{'image':50s} {'ratio':>9s} {'cls_diff':>10s} {'box_diff':>10s} {'pt':>3s} {'tf':>3s} {'IoU':>7s}")
    for p in imgs:
        im0 = load_image(p)
        H0, W0 = im0.shape[:2]
        r, pl, pt_ = letterbox_params(W0, H0)
        x = preprocess(im0)

        with torch.no_grad():
            o = m.model(torch.from_numpy(x[None].transpose(0, 3, 1, 2)))
        rp = (o[0] if isinstance(o, (list, tuple)) else o).cpu().numpy()
        rt = run_tflite(px_it, px_IN, px_OUT, x)

        # 1. convention: ratio of pt boxes to tflite boxes
        pb, tb = rp[0, :4].astype(np.float64), rt[0, :4].astype(np.float64)
        mask = np.abs(tb) > 1e-6
        ratio = float(np.median(pb[mask] / tb[mask]))
        box_diff = float(np.abs(pb - tb).max())
        # 2. class rows untouched
        cls_diff = float(np.abs(rp[0, 4:] - rt[0, 4:]).max())
        # 3. detections agree, both decoded at scale=1.0
        dpt = decode(rp, r, pl, pt_, 1.0)
        dtf = decode(rt, r, pl, pt_, 1.0)
        classes_match = [d[0] for d in dpt] == [d[0] for d in dtf]
        top_iou = round(iou(dpt[0][2], dtf[0][2]), 4) if (dpt and dtf) else None
        min_iou = (
            round(min(iou(a[2], b[2]) for a, b in zip(dpt, dtf)), 4) if (dpt and dtf and len(dpt) == len(dtf)) else None
        )

        rows.append(
            dict(
                image=os.path.basename(p),
                box_ratio_median=round(ratio, 4),
                box_max_absdiff=box_diff,
                cls_max_absdiff=cls_diff,
                pt_dets=len(dpt),
                tflite_dets=len(dtf),
                count_match=len(dpt) == len(dtf),
                classes_match=classes_match,
                top1_iou=top_iou,
                min_iou=min_iou,
            )
        )
        print(
            f"{os.path.basename(p):50s} {ratio:9.4f} {cls_diff:10.2e} {box_diff:10.2e} "
            f"{len(dpt):3d} {len(dtf):3d} {str(min_iou):>7s}"
        )

    # Control: the original model must still be normalized, i.e. the two models
    # must be distinguishable at runtime. Otherwise "we changed it" is unfalsifiable.
    control = None
    if nm:
        im0 = load_image(imgs[0])
        x = preprocess(im0)
        rn = run_tflite(nm[0], nm[1], nm[2], x)
        control = dict(
            normalized_box_rows_max=float(rn[0, :4].max()),
            pixelbox_box_rows_max=float(run_tflite(px_it, px_IN, px_OUT, x)[0, :4].max()),
        )
        print(
            f"\ncontrol: original box-rows max = {control['normalized_box_rows_max']:.4f} (normalized, ~1) | "
            f"pixelbox box-rows max = {control['pixelbox_box_rows_max']:.2f} (pixels, ~640)"
        )

    checks = {
        "convention_flipped": all(abs(r["box_ratio_median"] - 1.0) < 0.01 for r in rows),
        "class_rows_unchanged": all(r["cls_max_absdiff"] < 1e-4 for r in rows),
        "boxes_match_pt": all(r["box_max_absdiff"] < 1e-2 for r in rows),
        "det_counts_match": all(r["count_match"] for r in rows),
        "classes_match": all(r["classes_match"] for r in rows),
        "iou_ok": all(r["min_iou"] is None or r["min_iou"] >= 0.99 for r in rows),
    }
    if control:
        # Deliberately loose. Boxes may extend past the letterbox edge, so a
        # normalized row legitimately exceeds 1.0 (measured: 1.0017) -- which is
        # why TFLITE_CONTRACT.md v1's `assert max(rows 0..3) <= 1.001` was a
        # false-alarm waiting to happen. The two conventions sit ~640x apart, so
        # any threshold in between separates them without hair-splitting.
        checks["models_distinguishable"] = (
            control["normalized_box_rows_max"] < 10.0 and control["pixelbox_box_rows_max"] > 100.0
        )

    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"checks": checks, "control": control, "per_image": rows}, indent=2))
        print(f"wrote {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
