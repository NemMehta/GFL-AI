import os
import torch
from pathlib import Path
from azure.storage.blob import ContainerClient
from ultralytics import YOLO

# -------------------------------------------------------
# PERFORMANCE FLAGS
# -------------------------------------------------------
torch.backends.cudnn.benchmark = True

# -------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------
BATCH = 512
EPOCHS = 100
IMGSZ = 640

num_workers = min(16, max(8, (os.cpu_count() or 4)))

# -------------------------------------------------------
# AZURE BLOB CONFIG
# -------------------------------------------------------
AZURE_CONFIG = {
    "ConnectionString": "https://gflstorageblob.blob.core.windows.net",
    "Container": "dotnetbackend-container",
    "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2026-01-31T16:03:23Z&st=2025-12-03T07:48:23Z&spr=https&sig=XeBVuCefwPKVMc2loUsAvi8bXU1klfEi89jmT1fZBjI%3D"
}

BLOB_PREFIX = "Practice"
LOCAL_DATA_DIR = Path("/mnt/dataset")  # 🔥 FAST LOCAL NVMe

# -------------------------------------------------------
# DOWNLOAD DATASET FROM BLOB
# -------------------------------------------------------
def download_dataset():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    container_client = ContainerClient(
        account_url=AZURE_CONFIG["ConnectionString"],
        container_name=AZURE_CONFIG["Container"],
        credential=AZURE_CONFIG["SasToken"]
    )

    blobs = container_client.list_blobs(name_starts_with=BLOB_PREFIX)

    for blob in blobs:
        local_path = LOCAL_DATA_DIR / blob.name.replace(BLOB_PREFIX, "").lstrip("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as f:
            f.write(container_client.download_blob(blob.name).readall())

    print("✅ Dataset downloaded from Azure Blob")

# -------------------------------------------------------
# YOLO TRAIN
# -------------------------------------------------------
def main():
    download_dataset()

    dataset_yaml = LOCAL_DATA_DIR / "dataset.yaml"
    MODELS_DIR = Path("./models")
    MODELS_DIR.mkdir(exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    start_from = "yolov8m.pt"  # 🔥 GPU-friendly

    model = YOLO(start_from)

    model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        auto_batch=False,
        project=str(MODELS_DIR),
        name="tmp_global",
        exist_ok=True,
        device=device,
        workers=num_workers,
        cache=True,
        amp=True,
        save=True,
        verbose=False,
    )

if __name__ == "__main__":
    main()
