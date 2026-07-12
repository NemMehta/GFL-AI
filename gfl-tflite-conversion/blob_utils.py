import os
from azure.storage.blob import BlobServiceClient

from dotenv import load_dotenv
load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")

blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container = blob_service.get_container_client(CONTAINER_NAME)


# def list_pt_files():
#     pt_files = []
#     for blob in container.list_blobs(name_starts_with="models/"):
#         if blob.name.endswith(".pt"):
#             pt_files.append(blob.name)
#     return pt_files

#------------------updated-----------------#
def get_latest_pt_blob():
    """
    Returns the latest .pt blob from models/ using Azure blob last_modified timestamp.
    """
    latest_blob = None

    for blob in container.list_blobs(name_starts_with="models/"):
        if blob.name.endswith(".pt"):
            if latest_blob is None or blob.last_modified > latest_blob.last_modified:
                latest_blob = blob

    return latest_blob.name if latest_blob else None


def download_blob(blob_path, local_path):
    with open(local_path, "wb") as file:
        file.write(container.download_blob(blob_path).readall())


def upload_blob(local_path, blob_path):
    with open(local_path, "rb") as file:
        container.upload_blob(blob_path, file, overwrite=True)




def move_existing_tflite():
    """
    Correct method: Moves existing tflite/*.tflite files to tflite_previous/
    using start_copy_from_url(), then deletes the originals.
    """
    for blob in container.list_blobs(name_starts_with="tflite/"):
        if blob.name.endswith(".tflite"):

            source_blob_name = blob.name
            target_blob_name = blob.name.replace("tflite/", "tflite_previous/")

            source_blob = container.get_blob_client(source_blob_name)
            target_blob = container.get_blob_client(target_blob_name)

            # Get the full URL required for Start Copy
            source_url = source_blob.url

            print(f"[move_existing_tflite] Copying {source_blob_name} → {target_blob_name}")

            # Start async server-side copy
            target_blob.start_copy_from_url(source_url)

            # After copy completes, delete original blob
            container.delete_blob(source_blob_name)

            print(f"[move_existing_tflite] Moved to tflite_previous/: {target_blob_name}")

