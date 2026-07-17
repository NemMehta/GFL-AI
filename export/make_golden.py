"""Build the golden bundle the Android team bisects against.

Deliberately imports `decode` from verify_tflite.py -- the golden is produced by
the exact code the recipient runs, so the bundle and its checker cannot drift.

The v1 golden used bus.jpg, which contains no fish: final_detections.json was
empty at the real conf=0.25, so the only usable comparison set was a
conf=0.001 file full of 0.08-score noise. This regenerates from a real fish
image so the golden is non-empty at the threshold that actually ships.

Usage:
  python export/make_golden.py --model <pixelbox.tflite> --image <fish.webp> \
      --names <Names.txt> --out <golden_dir>
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from verify_tflite import CONF, IOU, MAX_DET, decode, detect_convention

IMGSZ = 640


def load_image(p):
    im = cv2.imread(str(p))
    if im is None:
        from PIL import Image

        im = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im


def letterbox(im0, imgsz=IMGSZ):
    """Ultralytics LetterBox, reimplemented so the bundle documents itself."""
    H0, W0 = im0.shape[:2]
    r = min(imgsz / W0, imgsz / H0)
    nW, nH = round(W0 * r), round(H0 * r)
    resized = cv2.resize(im0, (nW, nH), interpolation=cv2.INTER_LINEAR)
    dw, dh = (imgsz - nW) / 2, (imgsz - nH) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    x = np.ascontiguousarray(padded[:, :, ::-1], np.float32) / 255.0  # BGR->RGB, /255
    meta = dict(
        r=round(r, 6), pad_left=left, pad_right=right, pad_top=top, pad_bottom=bottom, W0=W0, H0=H0
    )
    return x, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--names", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    names = args.names.read_text(encoding="utf-8").splitlines()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    im0 = load_image(args.image)
    x, meta = letterbox(im0)
    xb = x[None]

    it = tf.lite.Interpreter(model_path=str(args.model))
    it.allocate_tensors()
    IN, OUT = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(IN["index"], xb.astype(IN["dtype"]))
    it.invoke()
    raw = it.get_tensor(OUT["index"])

    conv, mx = detect_convention(raw)
    scale = 1.0 if conv == "pixel" else 640.0
    print(f"convention: {conv} (box-rows max {mx:.3f}) -> scale x{scale:g}")

    dets = decode(raw, meta["r"], meta["pad_left"], meta["pad_top"], scale, conf_th=CONF)
    print(f"detections @ conf={CONF}: {len(dets)}")
    if not dets:
        raise SystemExit(
            "golden would be EMPTY at conf=0.25 -- pick an image with a confident fish. "
            "That emptiness is exactly the v1 bus.jpg problem."
        )

    # cross-check our numpy NMS against torchvision, so the bundle's numbers are
    # not merely self-consistent but agree with the reference implementation.
    try:
        import torch, torchvision

        p = raw[0]
        b = p[:4].astype(np.float64) * scale
        s = p[4:]
        cls, cf = s.argmax(0), s.max(0)
        k = cf > CONF
        B = np.stack([b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2], 1)[k]
        idx = torchvision.ops.batched_nms(
            torch.tensor(B, dtype=torch.float32),
            torch.tensor(cf[k], dtype=torch.float32),
            torch.tensor(cls[k]),
            IOU,
        ).numpy()[:MAX_DET]
        tv_classes = sorted(int(cls[k][i]) for i in idx)
        np_classes = sorted(d[0] for d in dets)
        assert tv_classes == np_classes, f"numpy NMS != torchvision NMS: {np_classes} vs {tv_classes}"
        print(f"numpy NMS matches torchvision batched_nms ({len(idx)} kept)")
    except ImportError:
        print("torch not present -- skipped NMS cross-check")

    (out / "letterbox.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "input_tensor.npy", xb.astype(np.float32))
    np.save(out / "raw_output.npy", raw.astype(np.float32))

    payload = dict(
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        box_convention=conv,
        decode_scale=scale,
        source_image=args.image.name,
        detections=[
            dict(cls=c, cls_name=names[c] if c < len(names) else "??", conf=s, xyxy=xy)
            for c, s, xy in dets
        ],
    )
    (out / "final_detections.json").write_text(json.dumps(payload, indent=2))

    # top-20 pre-NMS candidates, for anchor-level bisection
    p = raw[0]
    s = p[4:]
    cf, cl = s.max(0), s.argmax(0)
    top = np.argsort(-cf)[:20]
    unit = "px" if conv == "pixel" else "norm"
    with open(out / "topk_anchors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", f"cx_{unit}", f"cy_{unit}", f"w_{unit}", f"h_{unit}", "cls_idx", "cls_name", "score"])
        for a in top:
            w.writerow(
                [int(a)] + [f"{p[i][a]:.6f}" for i in range(4)]
                + [int(cl[a]), names[cl[a]] if cl[a] < len(names) else "??", f"{cf[a]:.6f}"]
            )

    dst = out / f"image{args.image.suffix}"
    dst.write_bytes(args.image.read_bytes())

    print(f"\nwrote golden -> {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.stat().st_size:>12,}  {f.name}")
    print("\ntop detections:")
    for c, s_, xy in dets[:5]:
        print(f"  {s_:.3f}  [{c}] {names[c]}  {xy}")


if __name__ == "__main__":
    main()
