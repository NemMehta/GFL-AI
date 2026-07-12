



import logging
import yaml
from pathlib import Path
from azure.storage.blob import ContainerClient
from app_backend.config import TRAIN
from app_backend.config import AZURE_CONFIG  # Import the global Azure config

# Logger setup
logger = logging.getLogger(__name__)

# # Azure storage config
# AZURE_CONFIG = {
#     "ConnectionString": "https://gflstorageaccount.blob.core.windows.net/dotnetbackend-container",
#     "Container": "dotnetbackend-container",
#     # "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D",
#     "SasToken": "sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2025-12-31T15:38:34Z&st=2025-09-17T07:23:34Z&spr=https&sig=IOG7m4n1ZfNb8SnCOBBURLvSPYKR0qiGjToBH8AmUzs%3D"
# }


def sync_training_data_from_azure(local_root: Path):
    """
    Pull ONLY augmentation/training_data/data.yaml from Azure.
    Overwrites local file if Azure version is newer.
    """
    try:
        container = AZURE_CONFIG.get("Container")
        sas_token = AZURE_CONFIG.get("SasToken")
        base_url = AZURE_CONFIG.get("ConnectionString")

        if not (container and sas_token and base_url):
            raise ValueError("Azure config is incomplete")

        container_client = ContainerClient.from_container_url(
            f"{base_url}?{sas_token}"
        )

        target_blob = "augmentation/training_data/data.yaml"
        logger.info(f"🔍 Checking for blob: {target_blob}")

        blob_client = container_client.get_blob_client(target_blob)
        local_path = local_root / "data.yaml"

        # Ensure directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"📂 Local target path: {local_path}")

        # props = blob_client.get_blob_properties()
        try:
            props = blob_client.get_blob_properties()
        except Exception:
            logger.warning("⚠️ data.yaml not found on Azure — using local file instead.")
            return local_path

        azure_ts = props.last_modified.timestamp()
        needs_download = True

        if local_path.exists():
            local_ts = local_path.stat().st_mtime
            logger.debug(f"⏱ Local timestamp={local_ts}, Azure timestamp={azure_ts}")
            if local_ts >= azure_ts:
                needs_download = False
                logger.info("✅ Local data.yaml is already up-to-date. Skipping download.")

        if needs_download:
            with open(local_path, "wb") as f:
                stream = blob_client.download_blob()
                f.write(stream.readall())
            logger.info(f"⬇️ Downloaded {target_blob} → {local_path}")

        logger.info(f"✅ Synced dataset.yaml to: {local_path}")
        return local_path

    except Exception as e:
        logger.exception(f"❌ Failed to fetch training data.yaml from Azure: {e}")
        raise


def task1_run(dataset_root: Path = None):
    """
    Ensure only data.yaml is synced from Azure before training.
    Returns dict: {"names": [...], "class_map": {...}}
    """
    dataset_root = Path(dataset_root or TRAIN.DATASET_ROOT)
    data_yaml = dataset_root / "data.yaml"

    logger.info(f"🔍 Starting task1_run to parse dataset.yaml {data_yaml}")
    print(f"🔍 Starting task1_run to parse dataset.yaml {data_yaml}")

    logger.info(f"📂 Dataset root: {dataset_root}")
    logger.info(f"🔍 Looking for dataset.yaml at: {data_yaml}")

    try:
        # Always sync from Azure
        synced_path = sync_training_data_from_azure(dataset_root)

        if not synced_path.exists():
            logger.error("❌ data.yaml missing even after Azure sync.")
            raise RuntimeError("dataset.yaml missing after sync")

        # Load dataset.yaml fully
        try:
            with open(synced_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as read_err:
            logger.exception(f"❌ Failed reading YAML from {synced_path}: {read_err}")
            raise

        class_names = data.get("names", [])
        class_map = data.get("class_map", {})

        if not class_names:
            logger.warning(f"⚠️ No classes found in {synced_path}")

        if not class_map:
            logger.warning(f"⚠️ No class_map found in {synced_path}")

        logger.info(f"📊 Loaded data.yaml with {len(class_names)} classes: {class_names}")

        return {
            "names": class_names,
            "class_map": class_map,
        }

    except Exception as e:
        logger.exception(f"❌ task1_run failed: {e}")
        raise








