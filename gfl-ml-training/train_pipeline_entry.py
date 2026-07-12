




"""
train_pipeline_entry.py
-------------------------------------------------
Azure ML Global Fish Training Pipeline
Handles MULTI-SPECIES → ONE GLOBAL MODEL training.
Reads:
    TRAIN_SPECIES           (list of species)
    SPECIES_PROJECT_MAP     ({species: [projectIds]})

Steps:
  1. Augmentation (per species)
  2. Task1
  3. Task2
  4. Global dataset build
  5. YOLO global model training
  6. Upload model/json/yaml to Blob
  7. Insert DB rows (one for each species → project)
  8. Send API notifications
-------------------------------------------------
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from azure.storage.blob import BlobServiceClient

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("train_pipeline_entry.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Project root import
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import training modules
from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset
from app_backend.modules.fish_training.parse_dataset import task1_run
from app_backend.modules.fish_training.group_species import task2_run
from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
from app_backend.modules.fish_training.train_models import train_single_model
# from app_backend.modules.fish_training.db_utils import insert_model_record
# from app_backend.modules.fish_training.train_models import notify_training_result
from app_backend.config import AZURE_CONFIG


import os
import torch

# -------------------------------------------------
# GPU PERFORMANCE OPTIMIZATION (A100 / H100)
# -------------------------------------------------
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------
# Azure Blob Upload
# ----------------------------------------------
def upload_to_blob(local_file, blob_path, container, connection_url, sas_token):
    """Uploads a single file to Azure Blob."""
    try:
        blob_service = BlobServiceClient(account_url=connection_url, credential=sas_token)
        container_client = blob_service.get_container_client(container)

        with open(local_file, "rb") as data:
            container_client.upload_blob(name=blob_path, data=data, overwrite=True)

        url = f"{connection_url}/{container}/{blob_path}"
        logger.info(f"📤 Uploaded: {url}")
        return url

    except Exception as e:
        logger.exception(f"❌ Upload failed for {blob_path}: {e}")
        return None


# # ----------------------------------------------
# # MAIN PIPELINE (Multi-species)
# # ----------------------------------------------
# def main():
#     logger.info("🚀 Starting GLOBAL multi-species training pipeline")

#     # Read environment variables from Azure ML
#     raw_species = os.getenv("TRAIN_SPECIES", "[]")
#     raw_map = os.getenv("SPECIES_PROJECT_MAP", "{}")

#     species_list = json.loads(raw_species)
#     species_project_map = json.loads(raw_map)

#     logger.info(f"🐟 Species List: {species_list}")
#     logger.info(f"📦 Species → Projects: {species_project_map}")

#     try:
#         # -----------------------------------------------------
#         # STEP 1 — Augmentation per species
#         # -----------------------------------------------------
#         for sp in species_list:
#             logger.info(f"🎨 Augmenting species: {sp}")
#             asyncio.run(build_augmented_dataset(sp))

#         # -----------------------------------------------------
#         # STEP 2 — Task1
#         # -----------------------------------------------------
#         logger.info("⚙️ Running Task1")
#         task1_run()

#         # -----------------------------------------------------
#         # STEP 3 — Task2
#         # -----------------------------------------------------
#         logger.info("⚙️ Running Task2")
#         changed_groups = task2_run()

#         # -----------------------------------------------------
#         # STEP 4 — Build global dataset
#         # -----------------------------------------------------
#         logger.info("📂 Building unified dataset")
#         generate_filtered_group_dataset(changed_groups=changed_groups)

#         # -----------------------------------------------------
#         # STEP 5 — Train ONE Global Model
#         # -----------------------------------------------------
#         logger.info("🏋️ Training Global YOLO Model")
#         train_single_model()

#         # train_single_model saved:
#         #    model_url
#         #    json_url
#         #    yaml_url
#         # inside db + API per project automatically

#         logger.info("🎉 Pipeline complete!")

#     except Exception as e:
#         logger.exception(f"❌ Pipeline failed: {e}")
#         raise


def main():
    logger.info("🚀 Starting GLOBAL multi-species training pipeline")

    species_list = json.loads(os.getenv("TRAIN_SPECIES", "[]"))
    species_project_map = json.loads(os.getenv("SPECIES_PROJECT_MAP", "{}"))

    logger.info(f"🐟 Species List: {species_list}")
    logger.info(f"📦 Species → Projects: {species_project_map}")

    # IMPORTANT FIX — Export for train_models.py
    os.environ["SPECIES_PROJECT_MAP"] = json.dumps(species_project_map)

    try:
        # STEP 1 — Augmentation
        for sp in species_list:
            asyncio.run(build_augmented_dataset(sp))

        # STEP 2 — Task1
        task1_run()

        # STEP 3 — Task2
        changed_groups = task2_run()

        # STEP 4 — Build global dataset (NO args)
        generate_filtered_group_dataset()

        # STEP 5 — Train YOLO global model
        train_single_model()

        logger.info("🎉 Pipeline complete!")

    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        raise




# ----------------------------- Entrypoint -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-species training pipeline")

    parser.add_argument(
        "--species",
        nargs="*",
        help="Local test: species names (space-separated)"
    )

    parser.add_argument(
        "--project_id",
        nargs="*",
        type=int,
        help="Local test: project IDs (space-separated)"
    )

    args = parser.parse_args()

    # ----------------------------------------------------
    # CASE 1 — Local CLI test
    # ----------------------------------------------------
    if args.species and args.project_id:

        if len(args.species) != len(args.project_id):
            raise ValueError("Number of species must match number of project IDs.")

        # Build map: { speciesName: [project IDs] }
        species_map = {}
        for sp, pid in zip(args.species, args.project_id):
            species_map.setdefault(sp, []).append(pid)

        # Build flat species list
        species_list = args.species

        # Store into environment for the existing main()
        os.environ["TRAIN_SPECIES"] = json.dumps(species_list)
        os.environ["SPECIES_PROJECT_MAP"] = json.dumps(species_map)

        print("🔧 Running LOCAL multi-species test mode:")
        print("TRAIN_SPECIES =", os.environ["TRAIN_SPECIES"])
        print("SPECIES_PROJECT_MAP =", os.environ["SPECIES_PROJECT_MAP"])

        main()   # ← Run your existing full pipeline
        sys.exit(0)

    # ----------------------------------------------------
    # CASE 2 — Azure ML job (env variables already provided)
    # ----------------------------------------------------
    print("🌐 Running AZURE ML mode...")
    main()











# # Old work
# """
# train_pipeline_entry.py
# -------------------------------------------------
# Azure ML Global Fish Training Pipeline
# Handles MULTI-SPECIES → ONE GLOBAL MODEL training.
# Reads:
#     TRAIN_SPECIES           (list of species)
#     SPECIES_PROJECT_MAP     ({species: [projectIds]})

# Steps:
#   1. Augmentation (per species)
#   2. Task1
#   3. Task2
#   4. Global dataset build
#   5. YOLO global model training
#   6. Upload model/json/yaml to Blob
#   7. Insert DB rows (one for each species → project)
#   8. Send API notifications
# -------------------------------------------------
# """

# import os
# import sys
# import json
# import asyncio
# import logging
# from pathlib import Path
# from datetime import datetime
# from azure.storage.blob import BlobServiceClient

# # Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.FileHandler("train_pipeline_entry.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # Project root import
# ROOT_DIR = Path(__file__).resolve().parent
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

# # Import training modules
# from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset
# from app_backend.modules.fish_training.parse_dataset import task1_run
# from app_backend.modules.fish_training.group_species import task2_run
# from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
# from app_backend.modules.fish_training.train_models import train_single_model
# # from app_backend.modules.fish_training.db_utils import insert_model_record
# # from app_backend.modules.fish_training.train_models import notify_training_result
# from app_backend.config import AZURE_CONFIG


# # ----------------------------------------------
# # Azure Blob Upload
# # ----------------------------------------------
# def upload_to_blob(local_file, blob_path, container, connection_url, sas_token):
#     """Uploads a single file to Azure Blob."""
#     try:
#         blob_service = BlobServiceClient(account_url=connection_url, credential=sas_token)
#         container_client = blob_service.get_container_client(container)

#         with open(local_file, "rb") as data:
#             container_client.upload_blob(name=blob_path, data=data, overwrite=True)

#         url = f"{connection_url}/{container}/{blob_path}"
#         logger.info(f"📤 Uploaded: {url}")
#         return url

#     except Exception as e:
#         logger.exception(f"❌ Upload failed for {blob_path}: {e}")
#         return None


# # # ----------------------------------------------
# # # MAIN PIPELINE (Multi-species)
# # # ----------------------------------------------
# # def main():
# #     logger.info("🚀 Starting GLOBAL multi-species training pipeline")

# #     # Read environment variables from Azure ML
# #     raw_species = os.getenv("TRAIN_SPECIES", "[]")
# #     raw_map = os.getenv("SPECIES_PROJECT_MAP", "{}")

# #     species_list = json.loads(raw_species)
# #     species_project_map = json.loads(raw_map)

# #     logger.info(f"🐟 Species List: {species_list}")
# #     logger.info(f"📦 Species → Projects: {species_project_map}")

# #     try:
# #         # -----------------------------------------------------
# #         # STEP 1 — Augmentation per species
# #         # -----------------------------------------------------
# #         for sp in species_list:
# #             logger.info(f"🎨 Augmenting species: {sp}")
# #             asyncio.run(build_augmented_dataset(sp))

# #         # -----------------------------------------------------
# #         # STEP 2 — Task1
# #         # -----------------------------------------------------
# #         logger.info("⚙️ Running Task1")
# #         task1_run()

# #         # -----------------------------------------------------
# #         # STEP 3 — Task2
# #         # -----------------------------------------------------
# #         logger.info("⚙️ Running Task2")
# #         changed_groups = task2_run()

# #         # -----------------------------------------------------
# #         # STEP 4 — Build global dataset
# #         # -----------------------------------------------------
# #         logger.info("📂 Building unified dataset")
# #         generate_filtered_group_dataset(changed_groups=changed_groups)

# #         # -----------------------------------------------------
# #         # STEP 5 — Train ONE Global Model
# #         # -----------------------------------------------------
# #         logger.info("🏋️ Training Global YOLO Model")
# #         train_single_model()

# #         # train_single_model saved:
# #         #    model_url
# #         #    json_url
# #         #    yaml_url
# #         # inside db + API per project automatically

# #         logger.info("🎉 Pipeline complete!")

# #     except Exception as e:
# #         logger.exception(f"❌ Pipeline failed: {e}")
# #         raise


# def main():
#     logger.info("🚀 Starting GLOBAL multi-species training pipeline")

#     species_list = json.loads(os.getenv("TRAIN_SPECIES", "[]"))
#     species_project_map = json.loads(os.getenv("SPECIES_PROJECT_MAP", "{}"))

#     logger.info(f"🐟 Species List: {species_list}")
#     logger.info(f"📦 Species → Projects: {species_project_map}")

#     # IMPORTANT FIX — Export for train_models.py
#     os.environ["SPECIES_PROJECT_MAP"] = json.dumps(species_project_map)

#     try:
#         # STEP 1 — Augmentation
#         for sp in species_list:
#             asyncio.run(build_augmented_dataset(sp))

#         # STEP 2 — Task1
#         task1_run()

#         # STEP 3 — Task2
#         changed_groups = task2_run()

#         # STEP 4 — Build global dataset (NO args)
#         generate_filtered_group_dataset()

#         # STEP 5 — Train YOLO global model
#         train_single_model()

#         logger.info("🎉 Pipeline complete!")

#     except Exception as e:
#         logger.exception(f"❌ Pipeline failed: {e}")
#         raise




# # ----------------------------- Entrypoint -----------------------------
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Run multi-species training pipeline")

#     parser.add_argument(
#         "--species",
#         nargs="*",
#         help="Local test: species names (space-separated)"
#     )

#     parser.add_argument(
#         "--project_id",
#         nargs="*",
#         type=int,
#         help="Local test: project IDs (space-separated)"
#     )

#     args = parser.parse_args()

#     # ----------------------------------------------------
#     # CASE 1 — Local CLI test
#     # ----------------------------------------------------
#     if args.species and args.project_id:

#         if len(args.species) != len(args.project_id):
#             raise ValueError("Number of species must match number of project IDs.")

#         # Build map: { speciesName: [project IDs] }
#         species_map = {}
#         for sp, pid in zip(args.species, args.project_id):
#             species_map.setdefault(sp, []).append(pid)

#         # Build flat species list
#         species_list = args.species

#         # Store into environment for the existing main()
#         os.environ["TRAIN_SPECIES"] = json.dumps(species_list)
#         os.environ["SPECIES_PROJECT_MAP"] = json.dumps(species_map)

#         print("🔧 Running LOCAL multi-species test mode:")
#         print("TRAIN_SPECIES =", os.environ["TRAIN_SPECIES"])
#         print("SPECIES_PROJECT_MAP =", os.environ["SPECIES_PROJECT_MAP"])

#         main()   # ← Run your existing full pipeline
#         sys.exit(0)

#     # ----------------------------------------------------
#     # CASE 2 — Azure ML job (env variables already provided)
#     # ----------------------------------------------------
#     print("🌐 Running AZURE ML mode...")
#     main()





















# # old work
# """
# train_pipeline_entry.py
# -------------------------------------------------
# Root-level entry point for Azure ML job submission.
# Executes the full fish training pipeline:
#   1. Augmentation
#   2. Task1
#   3. Task2
#   4. Filtered dataset generation
#   5. Model training
#   6. Upload results to Azure Blob
# -------------------------------------------------
# Usage:
#     python train_pipeline_entry.py --species "Yellow Tail Snapper (Ocyurus chrysurus)" --project_id 66
# """

# import os
# import sys
# import asyncio
# import logging
# from pathlib import Path
# import argparse
# from datetime import datetime
# from azure.storage.blob import BlobServiceClient

# # --- Set up logging ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.FileHandler("train_pipeline_entry.log"),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
# logger = logging.getLogger(__name__)

# # --- Ensure project root is importable ---
# ROOT_DIR = Path(__file__).resolve().parent
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

# # --- Imports from app_backend modules ---
# try:
#     from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset
#     from app_backend.modules.fish_training.parse_dataset import task1_run
#     from app_backend.modules.fish_training.group_species import task2_run
#     from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
#     from app_backend.modules.fish_training.train_models import train_all_groups
#     from app_backend.config import AZURE_CONFIG
# except ImportError as e:
#     logger.error(f"❌ Import failed: {e}")
#     raise


# # ----------------------------- Azure Upload Helper -----------------------------
# def upload_folder_to_blob(local_folder, container_name, dest_prefix, connection_url, sas_token):
#     """
#     Upload all files from a local folder to Azure Blob Storage.
#     """
#     local_folder = Path(local_folder)
#     if not local_folder.exists():
#         print(f"⚠️ Folder not found: {local_folder}")
#         return

#     blob_service_client = BlobServiceClient(account_url=connection_url, credential=sas_token)
#     container_client = blob_service_client.get_container_client(container_name)

#     for root, _, files in os.walk(local_folder):
#         for file in files:
#             local_path = Path(root) / file
#             blob_path = f"{dest_prefix}/{local_path.relative_to(local_folder)}"
#             try:
#                 with open(local_path, "rb") as data:
#                     container_client.upload_blob(name=blob_path, data=data, overwrite=True)
#                 print(f"📤 Uploaded → {blob_path}")
#             except Exception as ex:
#                 print(f"❌ Failed to upload {blob_path}: {ex}")


# # ----------------------------- Main Pipeline -----------------------------
# def main(species_name: str, project_id: int):
#     """
#     Executes the full fish training pipeline sequentially.
#     """
#     logger.info(f"🐟 Starting full pipeline for species: {species_name}, project: {project_id}")

#     # Set env variable so train_models.py can access PROJECT_ID
#     os.environ["PROJECT_ID"] = str(project_id)

#     try:
#         # --- Step 1: Augment dataset ---
#         logger.info("🚀 Step 1: Building augmented dataset ...")
#         asyncio.run(build_augmented_dataset(species_name=species_name))
#         # if species_name is None:
#         #     logger.info("🌍 Species name is None — running full multi-species augmentation.")
#         #     asyncio.run(build_augmented_dataset(species_name=None))  # Process all species
#         # elif str(species_name).lower() == "all":
#         #     logger.info("🌍 'all' keyword detected — running full multi-species augmentation.")
#         #     asyncio.run(build_augmented_dataset(species_name=None))
#         # else:
#         #     asyncio.run(build_augmented_dataset(species_name=species_name))
#         logger.info("✅ Augmentation complete.")


#         # --- Step 2: Task 1 ---
#         logger.info("⚙️ Step 2: Running Task 1 ...")
#         task1_run()
#         logger.info("✅ Task 1 complete.")

#         # --- Step 3: Task 2 ---
#         logger.info("⚙️ Step 3: Running Task 2 ...")
#         changed_groups = task2_run()
#         logger.info("✅ Task 2 complete.")
		
# 		# # --- Step 4: Filtered dataset generation ---
# 		# logger.info("📂 Step 4: Generating filtered dataset ...")
# 		# generate_filtered_group_dataset(changed_groups=changed_groups)
# 		# logger.info("✅ Filtered dataset generation complete.")

#         # --- Step 4: Filtered dataset generation ---
#         logger.info("📂 Step 4: Generating filtered dataset ...")
# 		# changed_groups = task2_run()
#         generate_filtered_group_dataset(changed_groups=changed_groups)
#         logger.info("✅ Filtered dataset complete.")


#         # generate_filtered_group_dataset()
#         logger.info("✅ Filtered dataset complete.")

#         # --- Step 5: Training models ---
#         logger.info("🏋️ Step 5: Starting YOLO training for all groups ...")
#         train_all_groups()
#         logger.info("✅ Model training complete.")

#         # --- Step 6: Upload results to Azure Blob ---
#         logger.info("☁️ Uploading outputs to Azure Blob Storage ...")

#         connection_url = AZURE_CONFIG["ConnectionString"]
#         container = AZURE_CONFIG["Container"]
#         sas_token = AZURE_CONFIG["SasToken"]

#         timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S_UTC")
#         base_prefix = f"UI/JobSubmission/{timestamp}"

#         # Upload both key folders
#         upload_folder_to_blob("/mnt/azureml/outputs/dataset", container, f"{base_prefix}/dataset", connection_url, sas_token)
#         upload_folder_to_blob("/mnt/azureml/outputs/temp_data", container, f"{base_prefix}/temp_data", connection_url, sas_token)

#         logger.info(f"✅ Uploaded dataset and temp_data folders to Azure Blob at prefix: {base_prefix}")
#         logger.info("🎯 Pipeline finished successfully!")

#     except Exception as e:
#         logger.exception(f"❌ Pipeline failed: {e}")
#         raise


# # ----------------------------- Entrypoint -----------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run full fish training pipeline")
#     parser.add_argument("--species", required=True, help="Species name for training")
#     parser.add_argument("--project_id", required=True, type=int, help="Project ID for training")

#     args = parser.parse_args()
#     main(species_name=args.species, project_id=args.project_id)































# # Old work on 21/11/2025
# """
# train_pipeline_entry.py
# -------------------------------------------------
# Root-level entry point for Azure ML job submission.
# Executes the full fish training pipeline:
#   1. Augmentation
#   2. Task1
#   3. Task2
#   4. Filtered dataset generation
#   5. Model training
# -------------------------------------------------
# Usage:
#     python train_pipeline_entry.py --species "Yellow Tail Snapper (Ocyurus chrysurus)" --project_id 66
# """

# import os
# import sys
# import asyncio
# import logging
# from pathlib import Path
# import argparse

# # --- Set up logging ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     handlers=[
#         logging.FileHandler("train_pipeline_entry.log"),
#         logging.StreamHandler(sys.stdout),
#     ],
# )
# logger = logging.getLogger(__name__)

# # --- Make sure root is on sys.path so imports work ---
# ROOT_DIR = Path(__file__).resolve().parent
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))

# # --- Imports from your app_backend modules ---
# try:
#     from app_backend.modules.fish_training.augment_pipeline import build_augmented_dataset
#     from app_backend.modules.fish_training.parse_dataset import task1_run
#     from app_backend.modules.fish_training.group_species import task2_run
#     from app_backend.modules.fish_training.filter_dataset import generate_filtered_group_dataset
#     from app_backend.modules.fish_training.train_models import train_all_groups
# except ImportError as e:
#     logger.error(f"❌ Import failed: {e}")
#     raise


# def main(species_name: str, project_id: int):
#     """
#     Executes the full fish training pipeline sequentially.
#     """
#     logger.info(f"🐟 Starting full pipeline for species: {species_name}, project: {project_id}")

#     # Set env variable so train_models.py can access PROJECT_ID
#     os.environ["PROJECT_ID"] = str(project_id)

#     try:
#         # --- Step 1: Augment dataset ---
#         logger.info("🚀 Step 1: Building augmented dataset ...")
#         asyncio.run(build_augmented_dataset(species_name=species_name))
#         logger.info("✅ Augmentation complete.")

#         # --- Step 2: Task 1 ---
#         logger.info("⚙️ Step 2: Running Task 1 ...")
#         task1_run()
#         logger.info("✅ Task 1 complete.")

#         # --- Step 3: Task 2 ---
#         logger.info("⚙️ Step 3: Running Task 2 ...")
#         task2_run()
#         logger.info("✅ Task 2 complete.")

#         # --- Step 4: Filtered dataset generation ---
#         logger.info("📂 Step 4: Generating filtered dataset ...")
#         generate_filtered_group_dataset()
#         logger.info("✅ Filtered dataset complete.")

#         # --- Step 5: Training models ---
#         logger.info("🏋️ Step 5: Starting YOLO training for all groups ...")
#         train_all_groups()
#         logger.info("✅ Model training complete.")

#         logger.info("🎯 Pipeline finished successfully!")

#     except Exception as e:
#         logger.exception(f"❌ Pipeline failed: {e}")
#         raise


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run full fish training pipeline")
#     parser.add_argument("--species", required=True, help="Species name for training")
#     parser.add_argument("--project_id", required=True, type=int, help="Project ID for training")

#     args = parser.parse_args()
#     main(species_name=args.species, project_id=args.project_id)
