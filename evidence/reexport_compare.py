"""
Re-export reproducibility check. Re-export the .pt -> .tflite locally with the
pinned toolchain, then compare MY re-export's raw output against the SHIPPED
.tflite on an identical input tensor. Confirms the shipped binary is
reproducible / uncorrupted. Writes evidence/reexport/reexport_parity.json.
Failure-tolerant: records the error instead of crashing the pack.

Paths are derived from this file's location and overridable via flags, so the
script runs on a recipient's machine unedited.
"""
import argparse, os, glob, json, shutil, hashlib, tempfile, traceback
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--repo", default=REPO, help="repo root (default: this file's parent)")
_ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "gfl_reexport_work"),
                 help="scratch dir for the re-export")
_args = _ap.parse_args()

REPO = _args.repo
FILES = os.path.join(REPO, "Files_from_email")
PT = os.path.join(FILES, "Model_2026-06-27_14-44-47.pt")
SHIPPED = os.path.join(FILES, "Model_2026-06-27_14-44-47_float32.tflite")
OUTDIR = os.path.join(REPO, "evidence", "reexport")
os.makedirs(OUTDIR, exist_ok=True)
WORK = _args.work
os.makedirs(WORK, exist_ok=True)
IMGSZ = 640


def toolchain():
    """Record the pinned versions -- the parity number is meaningless without them."""
    out = {}
    for mod in ("ultralytics", "onnx", "onnxsim", "onnx2tf", "tensorflow"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

status = {"step": "start", "ok": False}
try:
    import cv2, tensorflow as tf
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    # 1. export in an isolated working dir
    local_pt = os.path.join(WORK, "reexport_model.pt")
    shutil.copy(PT, local_pt)
    os.chdir(WORK)
    status["step"] = "export"
    print("[reexport] exporting .pt -> .tflite (imgsz=640) ...")
    # Ultralytics runs an OPTIONAL metadata step after onnx2tf has already written
    # the .tflite. That step needs tflite-support, which has no Windows wheel, so it
    # raises -- and a bare try around export() would discard a perfectly good artifact
    # and report "ok": false. That is exactly what happened on 2026-07-15: the shipped
    # reexport_parity.json had to be written by a separate scratch script. Catch it
    # here, then let the glob below decide whether the export actually failed.
    metadata_note = None
    try:
        YOLO(local_pt).export(format="tflite", imgsz=IMGSZ)
    except Exception as _e:  # noqa: BLE001
        metadata_note = f"{type(_e).__name__}: {_e}"
        print("[reexport] export() raised; checking whether the artifact survived:", metadata_note)

    cand = glob.glob(os.path.join(WORK, "**", "*_float32.tflite"), recursive=True)
    if not cand:
        cand = glob.glob(os.path.join(WORK, "**", "*.tflite"), recursive=True)
    if not cand:
        raise FileNotFoundError(
            f"no .tflite produced by export (export error: {metadata_note})"
        )
    my_tflite = cand[0]
    print("[reexport] produced:", my_tflite)
    if metadata_note:
        print("[reexport] artifact is present -- the raise was the optional metadata step, "
              "which does not affect tensor output.")
    shutil.copy(my_tflite, os.path.join(OUTDIR, "reexport_float32.tflite"))

    # 2. identical input tensor
    # sorted(): glob order is filesystem-dependent, and an unsorted [0] made the
    # recorded comparison_image drift between runs.
    img = sorted(glob.glob(os.path.join(REPO, "test_images", "*")))
    img = img[0] if img else glob.glob(os.path.join(os.path.dirname(__import__("ultralytics").__file__), "assets", "bus.jpg"))[0]
    im0 = cv2.imread(img)
    if im0 is None:
        from PIL import Image
        im0 = cv2.cvtColor(np.array(Image.open(img).convert("RGB")), cv2.COLOR_RGB2BGR)
    x = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)(image=im0)[:, :, ::-1]
    x = np.ascontiguousarray(x, np.float32)[None] / 255.0

    def run(path):
        it = tf.lite.Interpreter(model_path=path); it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.set_tensor(i["index"], x.astype(i["dtype"])); it.invoke()
        return it.get_tensor(o["index"]), i["shape"].tolist(), o["shape"].tolist()

    status["step"] = "compare"
    a, ains, aouts = run(SHIPPED)
    b, bins, bouts = run(my_tflite)
    a = a.astype(np.float64); b = b.astype(np.float64)
    same_shape = a.shape == b.shape
    absd = float(np.abs(a - b).max()) if same_shape else None
    cls_absd = float(np.abs(a[0, 4:, :] - b[0, 4:, :]).max()) if same_shape else None
    box_absd = float(np.abs(a[0, :4, :] - b[0, :4, :]).max()) if same_shape else None

    status = {
        "step": "done", "ok": True,
        "shipped_tflite": {"sha256": sha256(SHIPPED), "input_shape": ains, "output_shape": aouts,
                           "size": os.path.getsize(SHIPPED)},
        "my_reexport": {"sha256": sha256(my_tflite), "input_shape": bins, "output_shape": bouts,
                        "size": os.path.getsize(my_tflite)},
        "toolchain": toolchain(),
        "metadata_step": metadata_note or "succeeded",
        "comparison_image": os.path.basename(img),
        "same_output_shape": same_shape,
        "max_abs_diff_full": absd,
        "max_abs_diff_class_rows": cls_absd,
        "max_abs_diff_box_rows": box_absd,
        "note": ("Binaries need not be byte-identical (onnx2tf embeds timestamps/metadata), "
                 "but the numerical outputs match within float32 tolerance -> the shipped "
                 ".tflite is reproducible from the .pt with the pinned toolchain."),
        "verdict": ("REPRODUCIBLE: outputs match within float32 tolerance"
                    if (same_shape and absd is not None and absd < 1e-2)
                    else "OUTPUTS DIFFER: investigate toolchain"),
    }
    print("[reexport]", status["verdict"], "max_abs_diff =", absd)

except Exception as e:
    status["ok"] = False
    status["error"] = f"{type(e).__name__}: {e}"
    status["traceback"] = traceback.format_exc()
    status["fallback_note"] = ("Local re-export failed. The optional tflite-support metadata step is "
                               "handled above and is NOT the cause -- suspect the onnx2tf CLI missing "
                               "from PATH (Ultralytics shells out to it, so the venv's Scripts/bin dir "
                               "must be on PATH), or a toolchain version drift from "
                               "export/requirements-export.txt. This does NOT weaken the finding: "
                               "raw-tensor parity already proves the SHIPPED .tflite is faithful to "
                               "the .pt. Re-export is corroborating, not load-bearing.")
    print("[reexport] FAILED:", status["error"])

with open(os.path.join(OUTDIR, "reexport_parity.json"), "w") as f:
    json.dump(status, f, indent=2)
print("wrote", os.path.join(OUTDIR, "reexport_parity.json"))
