"""Export a YOLOv8 .pt to .tflite with box coordinates in PIXEL space.

Attribution / licence
---------------------
`forward_pixelbox` below is a MODIFIED COPY of `Detect.forward` from Ultralytics
YOLO v8.1.0 (`ultralytics/nn/modules/head.py`), which is licensed **AGPL-3.0**
(https://ultralytics.com/license). The only modification is the removal of the
tflite/edgetpu box-renormalization branch, as described below. This file is
therefore a derivative work and inherits AGPL-3.0.

Why this exists
---------------
Ultralytics' TFLite export deliberately renormalizes box coordinates to [0,1].
See ultralytics/nn/modules/head.py, Detect.forward (v8.1.0, lines 63-72):

    dbox = self.decode_bboxes(box)                      # <- correct, pixel-space

    if self.export and self.format in ("tflite", "edgetpu"):
        # Precompute normalization factor to increase numerical stability
        # See https://github.com/ultralytics/ultralytics/issues/7371
        img_h = shape[2]
        img_w = shape[3]
        img_size = torch.tensor([img_w, img_h, img_w, img_h], ...).reshape(1, 4, 1)
        norm = self.strides / (self.stride[0] * img_size)
        dbox = dist2bbox(self.dfl(box) * norm, ...)     # <- OVERWRITES with /640

The pixel-space boxes are computed first and then thrown away. There is no flag
to disable this in 8.1.0. A decoder that does not multiply by 640 gets boxes
~640x too small, which is the on-device accuracy collapse this repo diagnosed.

We monkey-patch Detect.forward here rather than editing the installed head.py,
because a `pip install -r requirements.txt` would silently revert a file edit and
nobody would notice until boxes collapsed again.

The stability rationale in issue #7371 applies to INT8 quantization. This is an
FP32 export; verify_pixelbox.py checks the claim empirically against the .pt.
"""

import argparse
import glob
import os
import shutil
import sys
import traceback
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors


def forward_pixelbox(self, x):
    """Detect.forward with the tflite/edgetpu box renormalization removed.

    Byte-for-byte the v8.1.0 method except that lines 65-72 (the `/img_size`
    branch) are gone. The line 58 branch is KEPT -- it avoids TF FlexSplitV ops
    and is unrelated to coordinate scaling.
    """
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x

    shape = x[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
        self.shape = shape

    if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
        box = x_cat[:, : self.reg_max * 4]
        cls = x_cat[:, self.reg_max * 4 :]
    else:
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

    dbox = self.decode_bboxes(box)

    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y if self.export else (y, x)


forward_pixelbox._gfl_pixelbox = True


def apply_patch():
    """Install the patch and prove it took, so a silent miss can't ship."""
    original = Detect.forward
    if getattr(original, "_gfl_pixelbox", False):
        return
    # Guard: if upstream ever drops the normalization on its own, the patch is
    # pointless and its silent presence would be misleading.
    import inspect

    body = inspect.getsource(original)
    if "Precompute normalization factor" not in body:
        raise RuntimeError(
            "Detect.forward no longer contains the tflite normalization branch. "
            "This ultralytics version may not need patching -- re-verify before exporting."
        )
    Detect.forward = forward_pixelbox
    assert getattr(Detect.forward, "_gfl_pixelbox", False), "patch did not take"


def export(pt_path: Path, out_path: Path, imgsz: int, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    local_pt = workdir / pt_path.name
    if not local_pt.exists():
        shutil.copy2(pt_path, local_pt)

    model = YOLO(str(local_pt))

    head = model.model.model[-1]
    if not isinstance(head, Detect):
        raise RuntimeError(f"expected a Detect head, got {type(head).__name__}")

    # Ultralytics' post-export metadata step needs tflite-support, which has no
    # Windows wheel. It raises AFTER onnx2tf has already written the .tflite, so
    # a bare `except` here would throw away a perfectly good artifact -- exactly
    # the bug that made evidence/reexport_compare.py contradict its own JSON.
    export_error = None
    try:
        model.export(format="tflite", imgsz=imgsz)
    except Exception as e:  # noqa: BLE001
        export_error = e

    pattern = str(workdir / f"{local_pt.stem}_saved_model" / "*_float32.tflite")
    hits = sorted(glob.glob(pattern))
    if not hits:
        print("[export] FAILED -- no float32 .tflite produced", file=sys.stderr)
        if export_error is not None:
            traceback.print_exception(type(export_error), export_error, export_error.__traceback__)
        raise SystemExit(1)

    if export_error is not None:
        print(
            f"[export] NOTE: export() raised after producing the artifact: "
            f"{type(export_error).__name__}: {export_error}"
        )
        print("[export] This is the OPTIONAL metadata step. The .tflite itself is fine.")
        print("[export] Consequence: no embedded metadata (imgsz/class names) -- ship the "
              "Names_*.txt sidecar.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hits[0], out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True, type=Path, help="source .pt")
    ap.add_argument("--out", required=True, type=Path, help="destination .tflite")
    ap.add_argument("--imgsz", type=int, default=640, help="export size (default 640)")
    ap.add_argument("--workdir", type=Path, default=Path("export/_work"), help="scratch dir")
    args = ap.parse_args()

    if not args.pt.exists():
        raise SystemExit(f"no such .pt: {args.pt}")

    apply_patch()
    print(f"[export] patched Detect.forward -> pixel-space boxes (no /{args.imgsz})")

    out = export(args.pt, args.out, args.imgsz, args.workdir)
    size = out.stat().st_size
    print(f"[export] wrote {out}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
