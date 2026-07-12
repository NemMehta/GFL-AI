import os
import glob
import shutil
from datetime import datetime
from ultralytics import YOLO

from blob_utils import download_blob, upload_blob, move_existing_tflite


def convert_pt_to_tflite(pt_blob_path: str) -> str:
    """
    Convert a .pt model from Azure Blob to a TFLite float32 model and upload it.

    Behaviour:
    - Reads the .pt blob name to derive model_name.
    - Lets Ultralytics create its own <model_name>_saved_model folder.
    - Finds the *_float32.tflite inside that folder.
    - Uploads that float32 model to:
        - tflite/<model_name>_float32.tflite       (overwrite)
        - tflite_previous/<model_name>_float32_<timestamp>.tflite
    - Moves any existing tflite/ models to tflite_previous/ (using move_existing_tflite).
    - Returns the primary blob path for the new TFLite model.
    """

    # ---------------------------------------------
    # 1. Derive model name from blob path
    # ---------------------------------------------
    blob_filename = os.path.basename(pt_blob_path)  # e.g. global_model_2025-12-05_21-07-00.pt
    if blob_filename.lower().endswith(".pt"):
        model_name = blob_filename[:-3]             # remove .pt
    else:
        model_name = os.path.splitext(blob_filename)[0]

    print(f"[convert] Using model name: {model_name}")

    local_pt = model_name + ".pt"

    # ---------------------------------------------
    # 2. Download PT model locally
    # ---------------------------------------------
    print(f"[convert] Downloading PT from blob: {pt_blob_path} → {local_pt}")
    download_blob(pt_blob_path, local_pt)

    # ---------------------------------------------
    # 3. Run Ultralytics export
    #    This should create:
    #    <model_name>_saved_model/<model_name>_float32.tflite
    # ---------------------------------------------
    print("[convert] Running Ultralytics export(format='tflite') ...")
    model = YOLO(local_pt)
    model.export(format="tflite")  # DO NOT change; this is the original behavior

    # ---------------------------------------------
    # 4. Find the saved_model folder Ultralytics created
    #    Example: global_model_2025-12-05_21-07-00_saved_model
    # ---------------------------------------------
    expected_folder = f"{model_name}_saved_model"

    if os.path.isdir(expected_folder):
        export_folder = expected_folder
    else:
        # fallback if name slightly different
        candidates = [
            d for d in os.listdir(".")
            if d.startswith(model_name) and d.endswith("_saved_model") and os.path.isdir(d)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find Ultralytics saved_model folder for {model_name}"
            )
        export_folder = candidates[0]

    print(f"[convert] Found export folder: {export_folder}")

    # ---------------------------------------------
    # 5. Find ONLY the float32 TFLite file
    #    e.g. global_model_2025-12-05_21-07-00_float32.tflite
    # ---------------------------------------------
    float32_pattern = os.path.join(export_folder, "*_float32.tflite")
    float32_files = glob.glob(float32_pattern)

    if not float32_files:
        # fallback: any .tflite (but still prefer float32)
        all_tflites = glob.glob(os.path.join(export_folder, "*.tflite"))
        if not all_tflites:
            raise FileNotFoundError("No .tflite file found in Ultralytics export folder.")
        tflite_local_path = all_tflites[0]
        print("[convert] WARNING: No *_float32.tflite found, using first .tflite file.")
    else:
        tflite_local_path = float32_files[0]

    tflite_filename = os.path.basename(tflite_local_path)
    print(f"[convert] Using TFLite float32 file: {tflite_filename}")

    # ---------------------------------------------
    # 6. Prepare blob paths
    # ---------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Main/latest model (overwrite allowed)
    primary_blob_path = f"tflite/{tflite_filename}"

    # Previous/backups – keep timestamped copies
    backup_blob_path = f"tflite_previous/{model_name}_float32_{timestamp}.tflite"

    # ---------------------------------------------
    # 7. Move any existing tflite/ models to tflite_previous/
    # ---------------------------------------------
    print("[convert] Archiving existing tflite/ models to tflite_previous/ ...")
    move_existing_tflite()

    # ---------------------------------------------
    # 8. Upload new model (float32) to tflite/ and tflite_previous/
    # ---------------------------------------------
    print(f"[convert] Uploading new float32 model to blob: {primary_blob_path}")
    upload_blob(tflite_local_path, primary_blob_path)

    print(f"[convert] Uploading backup float32 model to blob: {backup_blob_path}")
    upload_blob(tflite_local_path, backup_blob_path)

    # # ---------------------------------------------
    # # 9. Cleanup local files and folder
    # # ---------------------------------------------
    # try:
    #     if os.path.exists(local_pt):
    #         os.remove(local_pt)
    #     if os.path.isdir(export_folder):
    #         shutil.rmtree(export_folder)
    # except Exception as cleanup_err:
    #     print(f"[convert] Cleanup warning: {cleanup_err}")
    
    # ---------------------------------------------------
    # 9. Cleanup local files (PT, saved_model folder, ONNX, NPY)
    # ---------------------------------------------------
    try:
        # Remove downloaded .pt file
        if os.path.exists(local_pt):
            os.remove(local_pt)

        # Remove Ultralytics saved_model directory
        if os.path.isdir(export_folder):
            shutil.rmtree(export_folder)

        # Remove any ONNX files generated
        for file in os.listdir("."):
            if file.endswith(".onnx"):
                os.remove(file)

        # Remove any npy files generated
        for file in os.listdir("."):
            if file.endswith(".npy"):
                os.remove(file)

    except Exception as cleanup_err:
        print(f"[convert] Cleanup warning: {cleanup_err}")


    # ---------------------------------------------
    # 10. Return the blob path for latest model
    # ---------------------------------------------
    return primary_blob_path
