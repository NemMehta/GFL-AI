GFL Fish Detector - TFLite (pixel-box build, 2026-07-16)
=========================================================
Model_2026-06-27_14-44-47_float32_pixelbox.tflite
  sha256 10ad8fccc977ad90fbc7b32d839302583d7d5c32e8da5c3602c426d0eb9df84b
labels.txt  - 339 lines, line i == class i

INPUT   [1, 640, 640, 3] float32, NHWC, RGB, values /255 -> [0,1].
        Letterbox to 640x640 preserving aspect ratio, pad value 114.
        Bind by tensor index (no signature defs). Keep r / pad_left / pad_top.

OUTPUT  [1, 343, 8400] float32.  343 = 4 box rows + 339 class rows.
        Flat index: element (row r, prediction a) is at  r * 8400 + a
        (NOT a * 343 + r -- this is not [1,8400,343]).

        Rows 0-3 : cx, cy, w, h in PIXELS, 640-space.  USE AS-IS.
                   >>> Do NOT multiply by 640. <<<
                   Boxes are already fully decoded (DFL/anchors applied).
        Rows 4-342: class scores, ALREADY sigmoid-activated. No objectness row.
                    Do not apply sigmoid or softmax again.

DECODE  x1 = cx - w/2 ; y1 = cy - h/2 ; x2 = cx + w/2 ; y2 = cy + h/2
        NMS: conf 0.25, IoU 0.7, PER-CLASS (class-agnostic off), max_det 300.
        Un-letterbox: x = (x - pad_left) / r ;  y = (y - pad_top) / r

LABELS  This build has NO embedded metadata -- you MUST use labels.txt.
        Do not use a hardcoded label array, and do not use the older
        Names_Model_*.txt: that file has a "class_map:" header line, so a naive
        readLines() shifts every species by one.

CHECK   assert output shape == [1, 343, 8400]
        assert max(rows 0..3) > 100     # pixels; if it is ~1.0 you have the OLD
                                        # normalized model and DO need the x640
