from fastapi import FastAPI
from converter import convert_pt_to_tflite
#from blob_utils import list_pt_files
from blob_utils import get_latest_pt_blob


app = FastAPI()


@app.get("/convert")
def start_conversion():
    """
    - Finds latest .pt model in blob 'models/'.
    - Converts it to TFLite float32.
    - Uploads to:
        - tflite/<model_name>_float32.tflite
        - tflite_previous/<model_name>_float32_<timestamp>.tflite
    - Waits for completion and returns the final tflite blob path.
    """

    # 1. List .pt files under models/
    pt_files = get_latest_pt_blob()
    if not pt_files:
        return {"error": "No .pt files found in blob 'models/' folder."}

    # 2. Choose latest PT (alphabetically / by naming convention)
    # pt_path = sorted(pt_files)[-1]

    try:
        # 3. Convert and upload – this CALL BLOCKS until done
        tflite_blob_path = convert_pt_to_tflite(pt_path)

        # 4. Return to caller
        return {
            "status": "success",
            "pt_source_blob": pt_path,
            "tflite_blob_path": tflite_blob_path
        }

    except Exception as e:
        return {
            "status": "failed",
            "pt_source_blob": pt_path,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


#"127.0.0.1",