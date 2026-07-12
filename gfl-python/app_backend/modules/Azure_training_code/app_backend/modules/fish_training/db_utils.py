import sqlite3
import logging
from pathlib import Path
from datetime import datetime

# adjust if your DB path is different
DB_PATH = "D:\\Rahul Puri Data\\Projects\\Project GFL\\fish_records.db"

# -------------------- Logger --------------------
logger = logging.getLogger(__name__)

# # -------------------- Insert Helper --------------------
# def insert_model_record(species_name, model_path, json_file):
#     """
#     Update all model records in the DB for a given species with the new model path + json file.
#     This ensures that if multiple project IDs share the same species/model group, 
#     all of them get updated when the model is retrained.
#     """

#     conn = None
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()

#         # Ensure table exists
#         c.execute("""
#             CREATE TABLE IF NOT EXISTS models (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 project_id INTEGER,
#                 species_name TEXT,
#                 model_path TEXT,
#                 json_file TEXT,
#                 created_at TEXT DEFAULT (datetime('now'))
#             )
#         """)

#         # Normalize paths
#         def normalize_path(p):
#             p = str(p)
#             if p.startswith("http://") or p.startswith("https://"):
#                 return p
#             return str(Path(p).resolve())

#         norm_model_path = normalize_path(model_path)
#         norm_json_file = normalize_path(json_file)
#         now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

#         # Update *all* records for this species
#         c.execute("""
#             UPDATE models
#             SET model_path = ?, json_file = ?, created_at = ?
#             WHERE species_name = ?
#         """, (norm_model_path, norm_json_file, now, species_name))

#         if c.rowcount > 0:
#             logger.info(
#                 f"🔄 Updated {c.rowcount} model records for species='{species_name}' "
#                 f"with model='{norm_model_path}', json='{norm_json_file}'"
#             )
#         else:
#             logger.warning(f"⚠️ No existing model records found for species='{species_name}'")

#         conn.commit()

#     except Exception as e:
#         logger.exception(
#             f"❌ Failed to update model records for species='{species_name}'"
#         )
#         raise

#     finally:
#         if conn:
#             conn.close()


# def insert_model_record(project_id, species_name, model_path, json_file, status="Training"):
#     """
#     Insert or update model records for a given project_id + species.
#     Ensures each project-species pair is tracked properly.
#     Now also tracks training status.
#     """
#     conn = None
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()

#         # Ensure table exists with `status` column
#         c.execute("""
#             CREATE TABLE IF NOT EXISTS models (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 project_id INTEGER,
#                 species_name TEXT,
#                 model_path TEXT,
#                 json_file TEXT,
#                 status TEXT,
#                 created_at TEXT DEFAULT (datetime('now'))
#             )
#         """)

#         # Normalize paths (local or Azure URLs)
#         def normalize_path(p):
#             p = str(p) if p else ""
#             if p.startswith("http://") or p.startswith("https://"):
#                 return p
#             return str(Path(p).resolve())

#         norm_model_path = normalize_path(model_path)
#         norm_json_file = normalize_path(json_file)
#         now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

#         # ✅ Update if record already exists, else insert new
#         c.execute("""
#             SELECT id FROM models WHERE project_id = ? AND species_name = ?
#         """, (project_id, species_name))
#         existing = c.fetchone()

#         if existing:
#             c.execute("""
#                 UPDATE models
#                 SET model_path = ?, json_file = ?, status = ?, created_at = ?
#                 WHERE project_id = ? AND species_name = ?
#             """, (norm_model_path, norm_json_file, status, now, project_id, species_name))
#             logger.info(f"🔄 Updated model for project={project_id}, species={species_name}, status={status}")
#         else:
#             c.execute("""
#                 INSERT INTO models (project_id, species_name, model_path, json_file, status, created_at)
#                 VALUES (?, ?, ?, ?, ?, ?)
#             """, (project_id, species_name, norm_model_path, norm_json_file, status, now))
#             logger.info(f"🆕 Inserted model for project={project_id}, species={species_name}, status={status}")

#         conn.commit()

#     except Exception as e:
#         logger.exception(f"❌ Failed to insert/update model for project={project_id}, species={species_name}")
#         raise

#     finally:
#         if conn:
#             conn.close()


# app_backend/modules/fish_training/db_utils.py
def insert_model_record(project_id, species_name, model_path, json_file):
    """
    Insert or update model records for a given project_id + species.
    Ensures each project-species pair is tracked properly.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Ensure table exists
        c.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                species_name TEXT,
                model_path TEXT,
                json_file TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Normalize paths
        def normalize_path(p):
            p = str(p)
            if p.startswith("http://") or p.startswith("https://"):
                return p
            return str(Path(p).resolve())

        norm_model_path = normalize_path(model_path)
        norm_json_file = normalize_path(json_file)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Update record if project_id + species exists, else insert new
        c.execute("""
            SELECT id FROM models WHERE project_id = ? AND species_name = ?
        """, (project_id, species_name))
        existing = c.fetchone()

        if existing:
            c.execute("""
                UPDATE models
                SET model_path = ?, json_file = ?, created_at = ?
                WHERE project_id = ? AND species_name = ?
            """, (norm_model_path, norm_json_file, now, project_id, species_name))
            logger.info(f"🔄 Updated model for project={project_id}, species={species_name}")
        else:
            c.execute("""
                INSERT INTO models (project_id, species_name, model_path, json_file, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, species_name, norm_model_path, norm_json_file, now))
            logger.info(f"🆕 Inserted model for project={project_id}, species={species_name}")

        conn.commit()

    except Exception as e:
        logger.exception(f"❌ Failed to insert/update model for project={project_id}, species={species_name}")
        raise

    finally:
        if conn:
            conn.close()











# import sqlite3
# from pathlib import Path
# from datetime import datetime

# # adjust if your DB path is different
# DB_PATH = "D:\\Rahul Puri Data\\Projects\\Project GFL\\fish_records.db"


# def insert_model_record(project_id, species_name, model_path, json_file):
#     """
#     Insert a trained model record into the models table.

#     Args:
#         project_id (int or str): Project ID from API payload
#         species_name (str): Species name used in training
#         model_path (str): Azure URL or local path to the trained model .pt file
#         json_file (str): Azure URL or local path to group_hashes_current.json
#     """
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()

#     c.execute("""
#         CREATE TABLE IF NOT EXISTS models (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             project_id INTEGER,
#             species_name TEXT,
#             model_path TEXT,
#             json_file TEXT,
#             created_at TEXT DEFAULT (datetime('now'))
#         )
#     """)

#     # ✅ Use URL directly if it's already one, otherwise resolve local path
#     def normalize_path(p):
#         p = str(p)
#         if p.startswith("http://") or p.startswith("https://"):
#             return p
#         return str(Path(p).resolve())

#     c.execute("""
#         INSERT INTO models (project_id, species_name, model_path, json_file, created_at)
#         VALUES (?, ?, ?, ?, ?)
#     """, (
#         int(project_id) if str(project_id).isdigit() else None,
#         species_name,
#         normalize_path(model_path),
#         normalize_path(json_file),
#         datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
#     ))

#     conn.commit()
#     conn.close()
#     print(f"💾 Inserted model record into DB for project {project_id}, species '{species_name}'")
