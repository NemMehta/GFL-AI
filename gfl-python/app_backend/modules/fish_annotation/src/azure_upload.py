
import logging
from pathlib import Path
from azure.storage.blob import ContainerClient

# -------------------- Logger --------------------
logger = logging.getLogger(__name__)

# -------------------- Upload Augmented File --------------------
async def upload_augmented_file_async(file_path: Path, config: dict, subfolder: str) -> str:
    """
    Upload an augmented file (image/label) into augmentation/augment_images/{subfolder}.
    Returns the public blob URL or 'exception ...' string on failure.
    """
    try:
        container = config.get("Container", "")
        sas_token = config["SasToken"]
        base_url = config["ConnectionString"]

        # blob_name = f"augmentation/augment_images/{subfolder}/{Path(file_path).name}"
        blob_name = f"augmentation/{subfolder}/{Path(file_path).name}"

        container_client = ContainerClient.from_container_url(f"{base_url}?{sas_token}")
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = f"{base_url}/{blob_name}"
        logger.info(f"✅ Uploaded augmented file: {file_path} → {blob_url}")
        return blob_url

    except Exception as ex:
        logger.exception(f"❌ Failed to upload augmented file {file_path}: {ex}")
        return f"exception {str(ex)}"


# -------------------- Upload General File --------------------
async def upload_file_async(species: str, is_handheld: bool, file_path: Path, config: dict, subfolder: str) -> str:
    """
    Upload a file to Azure Blob Storage inside a given species/handheld/subfolder path.
    Returns the public blob URL or 'exception ...' string on failure.
    """
    try:
        connection_string = f"{config['ConnectionString']}?{config['SasToken']}"
        container = config.get("Container", "")

        hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
        species_folder = species

        blob_name = f"{species_folder}/{hand_held_folder}/{subfolder}/{Path(file_path).name}"

        container_client = ContainerClient.from_container_url(connection_string)
        blob_client = container_client.get_blob_client(blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = f"{config['ConnectionString']}/{blob_name}"
        logger.info(f"✅ Uploaded file for species={species}, handheld={is_handheld} → {blob_url}")
        return blob_url

    except Exception as ex:
        logger.exception(f"❌ Failed to upload file {file_path}: {ex}")
        return f"exception {str(ex)}"

















# import os
# from pathlib import Path
# from azure.storage.blob import ContainerClient

# async def upload_augmented_file_async(file_path: Path, config: dict, subfolder: str) -> str:
 
#     try:
#         container = config.get("Container", "")
#         sas_token = config["SasToken"]
#         base_url = config["ConnectionString"]

#         # Final blob name -> subfolder structure + filename
#         blob_name = f"augmentation/augment_images/{subfolder}/{Path(file_path).name}"

#         # Initialize client
#         container_client = ContainerClient.from_container_url(
#             f"{base_url}?{sas_token}"
#         )
#         blob_client = container_client.get_blob_client(blob_name)

#         with open(file_path, "rb") as data:
#             blob_client.upload_blob(data, overwrite=True)

        

#         blob_url = f"{base_url}/{blob_name}"
#         # print("=================================")
#         # print("Uploaded augmented image:", file_path)
#         # print("base_url:", base_url)
#         # print("container:", container)
#         # print("blob_name:", blob_name)
#         # print(f"Uploaded augmented images to: {blob_url}")
#         # print("=================================")
#         return blob_url
#     except Exception as ex:
#         return f"exception {str(ex)}"



# async def upload_file_async(species: str, is_handheld: bool, file_path: Path, config: dict, subfolder: str) -> str:
#     """
#     Upload a file to Azure Blob Storage inside a given subfolder (Annotated / Un-Annotated).
#     Returns the public blob URL.
#     """
#     try:
#         connection_string = f"{config['ConnectionString']}?{config['SasToken']}"
#         container = config.get("Container", "")

#         hand_held_folder = "Hand-Held" if is_handheld else "Not-Hand-Held"
#         species_folder = species

#         blob_name = f"{species_folder}/{hand_held_folder}/{subfolder}/{Path(file_path).name}"

#         container_client = ContainerClient.from_container_url(
#             f"{config['ConnectionString']}?{config['SasToken']}"
#         )
#         blob_client = container_client.get_blob_client(blob_name)

#         with open(file_path, "rb") as data:
#             blob_client.upload_blob(data, overwrite=True)

#         # print("=================================")
#         # print("Uploaded file:", file_path)
#         # print("blob_name:", blob_name)
#         # print("container:", container)
#         # print("blob_client:", blob_client)
#         # print("Full blob URL:", f"{config['ConnectionString']}/{blob_name}")
#         # print("=================================")

#         return f"{config['ConnectionString']}/{blob_name}"
    
#     except Exception as ex:
#         return f"exception {str(ex)}"
