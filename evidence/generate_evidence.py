"""
End-to-end evidence pack for the YOLOv8 .pt -> .tflite coordinate-normalization
root cause. Feeds the SAME preprocessed image to both models and produces:
  - provenance.json               (hashes + versions + image list)
  - raw/parity_all.csv            (per-image class/box parity numbers)
  - raw/<best>_raw.npz            (raw pt+tf tensors for the strongest image)
  - plots/box_scatter.png         (box_tf vs box_pt -> slope ~640)
  - plots/ratio_hist.png          (box_pt/box_tf -> spike at 640)
  - plots/classscore_parity.png   (scores tf vs pt -> y=x, R^2~1)
  - plots/conf_sweep.png          (det count vs conf, pt vs tflite)
  - overlays/overlay_<img>.png     (.pt | tflite-correct | tflite-WRONG)
  - overlays/decode_compare_<img>.json  (IoU correct/wrong vs .pt)
  - model_vs_model/compare_all.csv

Run with the pinned venv. No app code touched.
"""
import os, sys, glob, json, csv, hashlib, traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import tensorflow as tf
import ultralytics

REPO = r"C:\Users\Nem Mehta\GFL-AI-Repos"
FILES = os.path.join(REPO, "Files_from_email")
EV = os.path.join(REPO, "evidence")
PT = os.path.join(FILES, "Model_2026-06-27_14-44-47.pt")
TFLITE = os.path.join(FILES, "Model_2026-06-27_14-44-47_float32.tflite")
NAMES_TXT = os.path.join(FILES, "Names_Model_2026-06-27_14-44-47.txt")
IMGSZ = 640
for sub in ("raw", "plots", "overlays", "model_vs_model"):
    os.makedirs(os.path.join(EV, sub), exist_ok=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def load_bgr(path):
    im = cv2.imread(path)
    if im is None:  # .webp / odd formats -> PIL fallback
        from PIL import Image
        im = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im

def lb_params(W0, H0, imgsz=IMGSZ):
    r = min(imgsz / W0, imgsz / H0)
    newW, newH = round(W0 * r), round(H0 * r)
    dw, dh = (imgsz - newW) / 2, (imgsz - newH) / 2
    return r, round(dw - 0.1), round(dh - 0.1)

def preprocess(im0, imgsz=IMGSZ):
    im = LetterBox((imgsz, imgsz), auto=False, stride=32)(image=im0)
    im = im[:, :, ::-1]                                   # BGR->RGB
    return np.ascontiguousarray(im, dtype=np.float32) / 255.0   # HWC 0..1

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ---- inputs ----
imgs = sorted(sum([glob.glob(os.path.join(REPO, "test_images", e))
                   for e in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")], []))
demo_mode = False
if not imgs:
    imgs = [glob.glob(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))[0]]
    demo_mode = True
print(f"{len(imgs)} image(s); demo_mode={demo_mode}")

# ---- models ----
m_pt = YOLO(PT); m_pt.model.eval()
names = m_pt.names
interp = tf.lite.Interpreter(model_path=TFLITE); interp.allocate_tensors()
IN, OUT = interp.get_input_details()[0], interp.get_output_details()[0]

def raw_pt(nhwc):
    nchw = np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))
    with torch.no_grad():
        o = m_pt.model(torch.from_numpy(nchw))
    return (o[0] if isinstance(o, (list, tuple)) else o).cpu().numpy()

def raw_tf(nhwc):
    interp.set_tensor(IN["index"], nhwc.astype(IN["dtype"])); interp.invoke()
    return interp.get_tensor(OUT["index"])

def decode(raw, r, pl, pt_, W0, H0, denorm=True, conf_th=0.25, iou_th=0.7, max_det=300):
    pred = raw[0]
    b = pred[:4, :].astype(np.float64); sc = pred[4:, :]
    cls = sc.argmax(0); conf = sc.max(0)
    s = IMGSZ if denorm else 1.0
    cx, cy, w, h = b[0]*s, b[1]*s, b[2]*s, b[3]*s
    x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    keep = conf > conf_th
    if keep.sum() == 0:
        return []
    B = np.stack([x1, y1, x2, y2], 1)[keep]; C = conf[keep]; K = cls[keep]
    import torchvision
    idx = torchvision.ops.batched_nms(torch.tensor(B, dtype=torch.float32),
                                      torch.tensor(C, dtype=torch.float32),
                                      torch.tensor(K), iou_th).numpy()[:max_det]
    out = []
    for i in idx:
        X1 = (B[i, 0]-pl)/r; Y1 = (B[i, 1]-pt_)/r
        X2 = (B[i, 2]-pl)/r; Y2 = (B[i, 3]-pt_)/r
        out.append({"cls": int(K[i]), "name": names[int(K[i])], "conf": float(C[i]),
                    "xyxy": [float(np.clip(X1, 0, W0)), float(np.clip(Y1, 0, H0)),
                             float(np.clip(X2, 0, W0)), float(np.clip(Y2, 0, H0))]})
    return sorted(out, key=lambda d: -d["conf"])

def ul_pred(im0, conf_th):
    r0 = m_pt.predict(im0, imgsz=IMGSZ, conf=conf_th, iou=0.7, max_det=300, verbose=False)[0]
    o = []
    if r0.boxes is not None:
        for bx in r0.boxes:
            o.append({"cls": int(bx.cls[0]), "name": names[int(bx.cls[0])], "conf": float(bx.conf[0]),
                      "xyxy": [float(v) for v in bx.xyxy[0].tolist()]})
    return sorted(o, key=lambda d: -d["conf"])

# ============================================================ main loop
box_pairs, cls_pairs = [], []      # aggregate for plots
parity_rows = []
best = {"img": None, "conf": -1, "raw_pt": None, "raw_tf": None}
per_image = {}

for path in imgs:
    name = os.path.basename(path)
    try:
        im0 = load_bgr(path); H0, W0 = im0.shape[:2]
        r, pl, pt_ = lb_params(W0, H0)
        nhwc = preprocess(im0)[None]
        rp, rt = raw_pt(nhwc), raw_tf(nhwc)

        # parity numbers
        cls_pt, cls_tf = rp[0, 4:, :].astype(np.float64), rt[0, 4:, :].astype(np.float64)
        box_pt, box_tf = rp[0, :4, :].astype(np.float64), rt[0, :4, :].astype(np.float64)
        cls_maxd = float(np.abs(cls_pt - cls_tf).max())
        cls_meand = float(np.abs(cls_pt - cls_tf).mean())
        m = np.abs(box_tf) > 1e-4
        ratio_med = float(np.median(box_pt[m] / box_tf[m]))
        slope = float((box_tf.ravel() @ box_pt.ravel()) / (box_tf.ravel() @ box_tf.ravel()))
        resid = float(np.abs(box_tf * IMGSZ - box_pt).max())
        maxconf = float(cls_tf.max())

        parity_rows.append({"image": name, "W0": W0, "H0": H0,
                            "cls_max_absdiff": cls_maxd, "cls_mean_absdiff": cls_meand,
                            "box_slope_tf_to_pt": round(slope, 4), "box_ratio_median": round(ratio_med, 4),
                            "box_resid_after_x640": resid, "max_class_score": round(maxconf, 4)})

        # samples for plots
        n = box_tf.size
        si = np.random.RandomState(0).choice(n, size=min(1500, n), replace=False)
        box_pairs.append((box_tf.ravel()[si], box_pt.ravel()[si]))
        nc = cls_tf.size
        sj = np.random.RandomState(1).choice(nc, size=min(1500, nc), replace=False)
        cls_pairs.append((cls_tf.ravel()[sj], cls_pt.ravel()[sj]))

        # choose demo conf: real 0.25 if it yields dets, else drop to reveal behavior
        ref = ul_pred(im0, 0.25); demo_conf = 0.25
        if not ref:
            for c in (0.10, 0.05, 0.02, 0.01):
                ref = ul_pred(im0, c)
                if ref: demo_conf = c; break

        correct = decode(rt, r, pl, pt_, W0, H0, denorm=True, conf_th=demo_conf)
        wrong = decode(rt, r, pl, pt_, W0, H0, denorm=False, conf_th=demo_conf)
        iou_correct = iou(correct[0]["xyxy"], ref[0]["xyxy"]) if (correct and ref) else None
        iou_wrong = iou(wrong[0]["xyxy"], ref[0]["xyxy"]) if (wrong and ref) else None
        per_image[name] = {"demo_conf": demo_conf, "n_ref": len(ref), "n_correct": len(correct),
                           "n_wrong": len(wrong), "iou_correct_vs_pt": iou_correct,
                           "iou_wrong_vs_pt": iou_wrong, "max_class_score": round(maxconf, 4)}
        with open(os.path.join(EV, "overlays", f"decode_compare_{name}.json"), "w") as f:
            json.dump({"image": name, "demo_conf": demo_conf,
                       "pt_reference": ref[:10], "tflite_correct": correct[:10],
                       "tflite_wrong_no_x640": wrong[:10],
                       "iou_correct_vs_pt": iou_correct, "iou_wrong_vs_pt": iou_wrong}, f, indent=2)

        # overlay 3-panel
        rgb = im0[:, :, ::-1]
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        def draw(a, dets, color, title):
            a.imshow(rgb); a.set_title(title, fontsize=12); a.axis("off")
            for d in dets[:10]:
                x1, y1, x2, y2 = d["xyxy"]
                a.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, lw=2, ec=color, fc="none"))
                a.text(x1, max(0, y1-4), f'{d["name"][:18]} {d["conf"]:.2f}',
                       color="white", fontsize=7, bbox=dict(fc=color, ec="none", pad=1))
        draw(ax[0], ref, "deepskyblue", f".pt reference (Ultralytics)  conf>{demo_conf}")
        draw(ax[1], correct, "lime", "TFLite decoded CORRECTLY (xywh x640)")
        draw(ax[2], wrong, "red", "TFLite decoded WRONG (no x640)")
        ttl = f"{name}  [{W0}x{H0}]" + ("   DEMO: bus.jpg (no fish)" if demo_mode else "")
        fig.suptitle(ttl, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(EV, "overlays", f"overlay_{name}.png"), dpi=110)
        plt.close(fig)

        if maxconf > best["conf"]:
            best = {"img": name, "conf": maxconf, "raw_pt": rp, "raw_tf": rt}
        print(f"  {name}: maxconf={maxconf:.3f} demo_conf={demo_conf} "
              f"IoU(correct)={iou_correct} IoU(wrong)={iou_wrong}")
    except Exception:
        print(f"  !! FAILED on {name}"); traceback.print_exc()

# ---- save raw for the strongest image ----
if best["img"]:
    np.savez_compressed(os.path.join(EV, "raw", f"{best['img']}_raw.npz"),
                        raw_pt=best["raw_pt"].astype(np.float32), raw_tf=best["raw_tf"].astype(np.float32))

# ---- parity CSV ----
with open(os.path.join(EV, "raw", "parity_all.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(parity_rows[0].keys())); w.writeheader(); w.writerows(parity_rows)

# ============================================================ plots
BT = np.concatenate([p[0] for p in box_pairs]); BP = np.concatenate([p[1] for p in box_pairs])
CT = np.concatenate([p[0] for p in cls_pairs]); CP = np.concatenate([p[1] for p in cls_pairs])
gslope = float((BT @ BP) / (BT @ BT))

plt.figure(figsize=(6, 6))
plt.scatter(BT, BP, s=4, alpha=0.3)
xs = np.linspace(BT.min(), BT.max(), 50)
plt.plot(xs, gslope * xs, "r-", lw=1.5, label=f"fit slope = {gslope:.2f}")
plt.plot(xs, 640 * xs, "g--", lw=1, label="slope = 640 (expected)")
plt.xlabel("TFLite box value (normalized)"); plt.ylabel(".pt box value (pixels)")
plt.title("Box coordinates: TFLite vs .pt  (slope = 640 => TFLite is normalized)")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(EV, "plots", "box_scatter.png"), dpi=120); plt.close()

mask = np.abs(BT) > 1e-3
plt.figure(figsize=(7, 4))
plt.hist((BP[mask] / BT[mask]), bins=200, range=(600, 680))
plt.axvline(640, color="g", ls="--", label="640")
plt.xlabel(".pt / TFLite box ratio"); plt.ylabel("count")
plt.title("Ratio of box values concentrates at 640"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(EV, "plots", "ratio_hist.png"), dpi=120); plt.close()

ss_res = np.sum((CP - CT) ** 2); ss_tot = np.sum((CP - CP.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
plt.figure(figsize=(6, 6))
plt.scatter(CT, CP, s=4, alpha=0.3)
lim = [min(CT.min(), CP.min()), max(CT.max(), CP.max())]
plt.plot(lim, lim, "r--", lw=1, label=f"y=x  (R^2={r2:.6f})")
plt.xlabel("TFLite class score"); plt.ylabel(".pt class score")
plt.title("Class scores are identical between .pt and TFLite"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(EV, "plots", "classscore_parity.png"), dpi=120); plt.close()

# ============================================================ model vs model + conf sweep
m_tf = YOLO(TFLITE)
sweep = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
mvm_rows = []
pt_counts = {c: 0 for c in sweep}; tf_counts = {c: 0 for c in sweep}
for path in imgs:
    name = os.path.basename(path); im0 = load_bgr(path)
    rpt = m_pt.predict(im0, imgsz=IMGSZ, conf=0.25, iou=0.7, verbose=False)[0]
    rtf = m_tf.predict(im0, imgsz=IMGSZ, conf=0.25, iou=0.7, verbose=False)[0]
    npt = 0 if rpt.boxes is None else len(rpt.boxes)
    ntf = 0 if rtf.boxes is None else len(rtf.boxes)
    # top-1 agreement + IoU
    def top(res):
        if res.boxes is None or len(res.boxes) == 0: return None
        i = int(res.boxes.conf.argmax())
        return int(res.boxes.cls[i]), float(res.boxes.conf[i]), [float(v) for v in res.boxes.xyxy[i].tolist()]
    tp, tt = top(rpt), top(rtf)
    same = (tp and tt and tp[0] == tt[0]); iou_top = iou(tp[2], tt[2]) if (tp and tt) else None
    mvm_rows.append({"image": name, "pt_dets@0.25": npt, "tflite_dets@0.25": ntf,
                     "top1_same_class": bool(same),
                     "top1_iou": round(iou_top, 4) if iou_top is not None else None,
                     "pt_top_conf": round(tp[1], 4) if tp else None,
                     "tflite_top_conf": round(tt[1], 4) if tt else None})
    for c in sweep:
        rp2 = m_pt.predict(im0, imgsz=IMGSZ, conf=c, iou=0.7, verbose=False)[0]
        rt2 = m_tf.predict(im0, imgsz=IMGSZ, conf=c, iou=0.7, verbose=False)[0]
        pt_counts[c] += 0 if rp2.boxes is None else len(rp2.boxes)
        tf_counts[c] += 0 if rt2.boxes is None else len(rt2.boxes)

with open(os.path.join(EV, "model_vs_model", "compare_all.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(mvm_rows[0].keys())); w.writeheader(); w.writerows(mvm_rows)

plt.figure(figsize=(7, 4))
plt.plot(sweep, [pt_counts[c] for c in sweep], "o-", label=".pt")
plt.plot(sweep, [tf_counts[c] for c in sweep], "s--", label="TFLite (Ultralytics decode)")
plt.xlabel("confidence threshold"); plt.ylabel("total detections (all images)")
plt.title("Detection count vs conf: .pt and TFLite overlap (correct decode)"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(EV, "plots", "conf_sweep.png"), dpi=120); plt.close()

# ============================================================ provenance
prov = {
    "generated_for": "GFL TFLite coordinate-normalization diagnostics",
    "demo_mode_no_fish_images": demo_mode,
    "images": [os.path.basename(p) for p in imgs],
    "artifacts": {"pt": os.path.basename(PT), "tflite": os.path.basename(TFLITE),
                  "names": os.path.basename(NAMES_TXT)},
    "sha256": {os.path.basename(PT): sha256(PT), os.path.basename(TFLITE): sha256(TFLITE),
               os.path.basename(NAMES_TXT): sha256(NAMES_TXT)},
    "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                 "tensorflow": tf.__version__, "numpy": np.__version__},
    "headline": {"global_box_slope_tf_to_pt": round(gslope, 4),
                 "class_score_r2": round(float(r2), 8),
                 "best_detection_image": best["img"], "best_max_class_score": round(best["conf"], 4)},
    "per_image": per_image,
}
with open(os.path.join(EV, "provenance.json"), "w") as f:
    json.dump(prov, f, indent=2)

# ============================================================ asserts
print("\n=== SELF-CONSISTENCY ASSERTS ===")
cls_ok = max(r["cls_max_absdiff"] for r in parity_rows) < 1e-4
slope_ok = 639 <= gslope <= 641
r2_ok = r2 > 0.999
print(f"class-row max abs diff < 1e-4 : {cls_ok}  (max {max(r['cls_max_absdiff'] for r in parity_rows):.2e})")
print(f"global box slope in [639,641] : {slope_ok}  ({gslope:.3f})")
print(f"class-score R^2 > 0.999       : {r2_ok}  ({r2:.6f})")
if not demo_mode:
    ic = [v["iou_correct_vs_pt"] for v in per_image.values() if v["iou_correct_vs_pt"] is not None]
    iw = [v["iou_wrong_vs_pt"] for v in per_image.values() if v["iou_wrong_vs_pt"] is not None]
    if ic: print(f"IoU(correct vs pt) min        : {min(ic):.3f}  (expect high)")
    if iw: print(f"IoU(wrong   vs pt) max        : {max(iw):.3f}  (expect ~0)")
print("\nALL CORE ASSERTS PASS" if (cls_ok and slope_ok and r2_ok) else "\n!! ASSERT FAILURE")
print("files:", sorted(os.listdir(EV)))
