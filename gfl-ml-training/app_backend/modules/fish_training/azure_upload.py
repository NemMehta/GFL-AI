import os
import logging
from pathlib import Path
from azure.storage.blob import ContainerClient

# Use global app logger instead of resetting logging here
logger = logging.getLogger(__name__)

# -------------------- Upload Augmented Files --------------------
async def upload_augmented_file_async(file_path: Path, config: dict, subfolder: str) -> str:
    try:
        container = config.get("Container", "")
        sas_token = config["SasToken"]
        base_url = config["ConnectionString"]

        # blob_name = f"augmentation/augment_images/{subfolder}/{Path(file_path).name}"
        blob_name = f"augmentation/{subfolder}/{Path(file_path).name}"

        container_client = ContainerClient.from_container_url(f"{base_url}/{container}?{sas_token}")
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = f"{base_url}/{blob_name}"
        logger.info(f"✅ Uploaded augmented file: {file_path} → {blob_url}")
        return blob_url

    except Exception:
        logger.exception(f"❌ Failed to upload augmented file {file_path}")
        return f"exception {file_path}"


# -------------------- Upload Training Files --------------------
async def upload_training_file_async(file_path: Path, config: dict, subfolder: str) -> str:
    try:
        container = config.get("Container", "")
        sas_token = config["SasToken"]
        base_url = config["ConnectionString"]

        # Special-case: dataset.yaml or data.yaml should go directly under training_data
        if file_path.name in {"data.yaml", "dataset.yaml"}:
            blob_name = f"augmentation/training_data/{file_path.name}"
        else:
            blob_name = f"augmentation/{subfolder}/{Path(file_path).name}"

        container_client = ContainerClient.from_container_url(f"{base_url}?{sas_token}")
        blob_client = container_client.get_blob_client(blob_name)

        logger.info(f"⬆️ Uploading training file: {file_path} → {blob_name}")

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = f"{base_url}/{blob_name}"
        logger.info(f"✅ Uploaded training file → {blob_url}")
        return blob_url

    except Exception:
        logger.exception(f"❌ Upload failed for training file {file_path}")
        return f"exception {file_path}"


# -------------------- Upload Model File --------------------
async def upload_model_file_async(file_path: Path, config: dict) -> str:
    """Upload trained model file directly under /models/ at Azure Blob Storage."""
    try:
        container_client = ContainerClient.from_container_url(
            f"{config['ConnectionString']}?{config['SasToken']}"
        )
        blob_name = f"models/{Path(file_path).name}"
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        url = f"{config['ConnectionString']}/{blob_name}"
        logger.info(f"☁️ Uploaded trained model to Azure: {url}")
        return url

    except Exception:
        logger.exception(f"❌ Failed to upload trained model {file_path}")
        return ""


# -------------------- Upload General File --------------------
async def upload_file_async(species: str, is_handheld: bool, file_path: Path, config: dict, subfolder: str) -> str:
    """Upload a file to Azure Blob Storage inside a species/handheld folder."""
    try:
        connection_string = f"{config['ConnectionString']}?{config['SasToken']}"
        hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
        species_folder = species

        blob_name = f"{species_folder}/{hand_held_folder}/{subfolder}/{Path(file_path).name}"
        container_client = ContainerClient.from_container_url(connection_string)
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = f"{config['ConnectionString']}/{blob_name}"
        logger.info(f"✅ Uploaded file (species={species}, handheld={is_handheld}) → {blob_url}")
        return blob_url

    except Exception:
        logger.exception(f"❌ Failed to upload file {file_path}")
        return f"exception {file_path}"





