# What changed to get correct boxes out of the .tflite

**Model:** `Model_2026-06-27_14-44-47_float32_pixelbox.tflite`, 2026-07-16.

The `.tflite` used to emit box coordinates **divided by 640**, while the source `.pt` emits them
in **pixels**. That is Ultralytics' export doing it deliberately — not a bug in our converter,
and not corruption. We removed that step at export time. The `.tflite` now emits pixel boxes and
matches the `.pt` exactly, so no `×640` is needed anywhere.

## Where the /640 came from

`ultralytics/nn/modules/head.py`, `Detect.forward` (v8.1.0). Line 63 computes the correct
pixel-space boxes — then lines 65-72 **overwrite** them:

```python
dbox = self.decode_bboxes(box)                     # <- correct, pixel-space

if self.export and self.format in ("tflite", "edgetpu"):
    # Precompute normalization factor to increase numerical stability
    # See https://github.com/ultralytics/ultralytics/issues/7371
    img_h = shape[2]
    img_w = shape[3]
    img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
    norm = self.strides / (self.stride[0] * img_size)
    dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)
```

Three things that explain why this went unnoticed for so long:

- The branch fires **only** for `tflite`/`edgetpu`. Every other export format keeps pixels —
  so the `.pt` and the `.tflite` disagreed while nothing looked wrong.
- **There is no flag to disable it** in 8.1.0.
- It is baked in at **ONNX-export time, before onnx2tf runs** — so the converter and onnx2tf were
  never at fault.

The stated reason is numerical stability, which matters for **INT8** quantization. This is an
FP32 export, so it buys us nothing.

## The change

A runtime monkey-patch in `export_pixelbox.py` — a copy of `Detect.forward` with lines 65-72
removed, and **nothing else touched**:

```python
def forward_pixelbox(self, x):
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
        box = x_cat[:, : self.reg_max * 4]      # KEPT: avoids TF FlexSplitV ops
        cls = x_cat[:, self.reg_max * 4 :]
    else:
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

    dbox = self.decode_bboxes(box)              # pixel-space; the /640 branch is gone

    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y if self.export else (y, x)

Detect.forward = forward_pixelbox
```

Then `YOLO(pt).export(format="tflite", imgsz=640)`.

Two deliberate choices:

- **The `saved_model/pb/tflite/edgetpu/tfjs` branch is kept.** It avoids TF `FlexSplitV` ops and
  has nothing to do with coordinate scaling. Only the normalization branch is dropped.
- **Monkey-patch, not an edit to `head.py`.** A `pip install -r requirements.txt` would silently
  revert a file edit, the boxes would collapse again, and nothing would signal it. The patch also
  checks the normalization branch is still present before applying — if a future Ultralytics drops
  it, the export fails loudly instead of lying.

## Proof it's faithful

We removed math from a graph, so "it looks right" isn't good enough. Measured against the source
`.pt` on the same letterboxed tensor, across all 7 test images (`pixelbox_parity.json`):

| Check | Result |
|---|---|
| Box coords vs `.pt` | median ratio **1.0000** (was `640.0`); max abs diff **3.5e-03** px |
| Class scores vs `.pt` | max abs diff **6.6e-06** — unchanged, as expected |
| Detection count & classes | **identical** on all 7 images (14 detections total) |
| Per-box IoU vs `.pt` | **1.0** |
| Old vs new, same image | box-rows max **1.0017** vs **641.12** |

The `.tflite` now agrees with the `.pt` with no convention delta. Class scores were never
affected by any of this.

---

Note: this changed the **export script only**. `gfl-tflite-conversion/converter.py` is untouched,
so the Azure conversion service still produces normalized models.
