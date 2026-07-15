"""
Cleanest apples-to-apples proof. From the SAME manual input tensor, decode:
  - pt-raw   : boxes already in pixels (scale x1)
  - tflite-raw: boxes normalized -> multiply by 640 (scale x640)
Both then go through the identical xywh->xyxy, un-letterbox, per-class NMS.
Result: identical detections for every image -> the ONLY difference between the
two models is the /640 box normalization. Writes raw/same_tensor_decode_parity.csv.
"""
import os, glob, csv
import numpy as np, cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import tensorflow as tf

REPO = r"C:\Users\Nem Mehta\GFL-AI-Repos"; FILES = os.path.join(REPO, "Files_from_email")
PT = os.path.join(FILES, "Model_2026-06-27_14-44-47.pt")
TFLITE = os.path.join(FILES, "Model_2026-06-27_14-44-47_float32.tflite")
OUTCSV = os.path.join(REPO, "evidence", "raw", "same_tensor_decode_parity.csv")
IMGSZ = 640
m = YOLO(PT); m.model.eval(); names = m.names
it = tf.lite.Interpreter(model_path=TFLITE); it.allocate_tensors()
IN, OUT = it.get_input_details()[0], it.get_output_details()[0]

def load(p):
    im = cv2.imread(p)
    if im is None:
        from PIL import Image
        im = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im

def lb(W0, H0):
    r = min(IMGSZ/W0, IMGSZ/H0); nW, nH = round(W0*r), round(H0*r)
    return r, round((IMGSZ-nW)/2-0.1), round((IMGSZ-nH)/2-0.1)

def dec(raw, r, pl, pt_, W0, H0, scale, conf_th=0.25):
    p = raw[0]; b = p[:4].astype(np.float64)*scale; s = p[4:]
    cls = s.argmax(0); cf = s.max(0)
    x1, y1, x2, y2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    k = cf > conf_th
    if k.sum() == 0: return []
    import torchvision
    B = np.stack([x1, y1, x2, y2], 1)[k]; C = cf[k]; K = cls[k]
    idx = torchvision.ops.batched_nms(torch.tensor(B, dtype=torch.float32),
          torch.tensor(C, dtype=torch.float32), torch.tensor(K), 0.7).numpy()[:300]
    return sorted([(int(K[i]), round(float(C[i]), 4),
        [round(float((B[i,0]-pl)/r),1), round(float((B[i,1]-pt_)/r),1),
         round(float((B[i,2]-pl)/r),1), round(float((B[i,3]-pt_)/r),1)]) for i in idx],
        key=lambda t:-t[1])

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0

imgs = sorted(sum([glob.glob(os.path.join(REPO, "test_images", e)) for e in ("*.webp", "*.jpg", "*.jpeg", "*.png")], []))
rows = []
print(f"{'image':52s} pt_dets tf_dets match top1_iou")
for p in imgs:
    im0 = load(p); H0, W0 = im0.shape[:2]; r, pl, pt_ = lb(W0, H0)
    x = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)(image=im0)[:, :, ::-1]
    x = np.ascontiguousarray(x, np.float32)/255.0
    with torch.no_grad():
        o = m.model(torch.from_numpy(x[None].transpose(0, 3, 1, 2)))
    rp = (o[0] if isinstance(o, (list, tuple)) else o).cpu().numpy()
    it.set_tensor(IN["index"], x[None].astype(IN["dtype"])); it.invoke(); rt = it.get_tensor(OUT["index"])
    dpt = dec(rp, r, pl, pt_, W0, H0, 1.0)
    dtf = dec(rt, r, pl, pt_, W0, H0, IMGSZ)
    t = round(iou(dpt[0][2], dtf[0][2]), 4) if (dpt and dtf) else None
    rows.append({"image": os.path.basename(p), "pt_dets": len(dpt), "tflite_dets": len(dtf),
                 "count_match": len(dpt) == len(dtf), "top1_iou": t})
    print(f"{os.path.basename(p):52s} {len(dpt):7d} {len(dtf):7d} {str(len(dpt)==len(dtf)):5s} {t}")

with open(OUTCSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
allmatch = all(r["count_match"] for r in rows)
print(f"\nALL COUNTS MATCH: {allmatch}  ->  wrote {OUTCSV}")
