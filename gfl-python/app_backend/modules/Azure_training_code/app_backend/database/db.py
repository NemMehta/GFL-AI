# import sqlite3
# from datetime import datetime
# from geopy.distance import geodesic
# import logging

# DB_PATH = "fish_records.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS fish_records (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             species TEXT,
#             length REAL,
#             latitude REAL,
#             longitude REAL,
#             timestamp TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# def insert_fish(species, length, latitude, longitude, timestamp):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute('''
#         INSERT INTO fish_records (species, length, latitude, longitude, timestamp)
#         VALUES (?, ?, ?, ?, ?)
#     ''', (species, length, latitude, longitude, timestamp))
#     conn.commit()
#     conn.close()


# def is_duplicate(species, length, latitude, longitude, timestamp_str):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     # Order by timestamp descending (latest first)
#     c.execute("SELECT * FROM fish_records WHERE species = ? ORDER BY timestamp DESC", (species,))
    
#     records = c.fetchall()
#     print(records)
#     conn.close()

#     timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
#     msg = "No duplicate found in any condition"
#     for record in records:
#         _, _, db_length, db_lat, db_lon, db_ts = record
#         db_time = datetime.strptime(db_ts, "%Y-%m-%d %H:%M:%S")

#         # Calculations
#         length_diff_ratio = abs(db_length - length) / db_length
#         geo_dist_km = geodesic((db_lat, db_lon), (latitude, longitude)).km
#         time_diff_sec = (timestamp - db_time).total_seconds()
#         # print("timestamp:", timestamp)
#         # print("db_time:", db_time)
#         # print("(timestamp - db_time)", (timestamp - db_time))
#         # print("(timestamp - db_time).days", (timestamp - db_time).days)
#         # print("abs((timestamp - db_time).days)", abs((timestamp - db_time).days))
#         days_diff = abs((timestamp - db_time).days)


#         # Same coordinates (within 10m), time < 10 minutes, and length is exactly the same → Likely duplicate
#         if geo_dist_km < 0.01 and abs(time_diff_sec) < 600 and length_diff_ratio < 0.10:
#             msg = f"Duplicate detected — same location (<10m) {geo_dist_km*1000:.2f} meters, time <10min ({time_diff_sec}s), length match"
#             return True, msg
        
#         # Length similarity check (±10%)
#         print("length_diff_ratio", length_diff_ratio)
#         # logging.info(f"length_diff_ratio {length_diff_ratio}")
#         if length_diff_ratio > 0.10:
#             msg = f"Skipped due to length difference: {length_diff_ratio:.2%} (DB: {db_length}, Input: {length})"
#             continue

#         # Time difference > 1 day (24 hours) → Always unique
#         print("days_diff", days_diff)
#         # logging.info(f"days_diff {days_diff}")
#         if days_diff >= 1:
#             msg = f"Unique due to time difference > 1 day: {days_diff} days"
#             continue
        
#         # Geolocation proximity check (within 200 meters)
#         print(f"geo_dist_km {geo_dist_km*1000:.2f}")
#         # logging.info(f"geo_dist_km {geo_dist_km*1000:.2f}")
#         if geo_dist_km > 0.2:
#             msg = f"Skipped due to distance > 200m: {geo_dist_km*1000:.2f} meters"
#             continue

#         # Distance < 200m, Time > 3 hours → Unique
#         print(f"time_diff_sec {time_diff_sec/60:.2f} minutes")
#         # logging.info(f"time_diff_sec {time_diff_sec/60:.2f} minutes")
#         if geo_dist_km <= 0.2 and time_diff_sec > 10800:
#             msg = f"Unique due to time > 3 hours at same location: {time_diff_sec/3600:.2f} hours"
#             continue

#     return False, msg

# if __name__ == "__main__":
#     init_db()



# import sqlite3
# from datetime import datetime
# from geopy.distance import geodesic
# import logging
# import os


# DB_PATH = "D:\\Rahul Puri Data\\Projects\\Project GFL\\fish_records.db"
# DB_PATH_DATA_COLLECTION = "D:\\Rahul Puri Data\\Projects\\Project GFL\\fish_data.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()

#     # Known species
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS fish_records (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             species TEXT,
#             length REAL,
#             latitude REAL,
#             longitude REAL,
#             timestamp TEXT,
#             image_path TEXT
#         )
#     ''')

#     # Unknown species
#     c.execute('''
#         CREATE TABLE IF NOT EXISTS unknown_fish (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             species TEXT,
#             length REAL,
#             latitude REAL,
#             longitude REAL,
#             timestamp TEXT,
#             image_path TEXT
#         )
#     ''')

#     conn.commit()
#     conn.close()

# def init_db_for_data_collection():
#     """Initialize database and base folder"""
#     conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
#     cursor = conn.cursor()
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS species (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             species_name TEXT UNIQUE NOT NULL
#         )
#     """)
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS images (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             image_path TEXT NOT NULL,
#             handheld BOOLEAN NOT NULL,
#             species_id INTEGER NOT NULL,
#             FOREIGN KEY(species_id) REFERENCES species(id)
#         )
#     """)
#     conn.commit()
#     conn.close()
#     # os.makedirs(DB_PATH_DATA_COLLECTION, exist_ok=True)
#     print("✅ DB initialized")

# def insert_unknown_fish(species, length, latitude, longitude, timestamp, image_path):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute('''
#         INSERT INTO unknown_fish (species, length, latitude, longitude, timestamp, image_path)
#         VALUES (?, ?, ?, ?, ?, ?)
#     ''', (species, length, latitude, longitude, timestamp, image_path))
#     conn.commit()
#     conn.close()

# def insert_fish(species, length, latitude, longitude, timestamp, image_path):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute('''
#         INSERT INTO fish_records (species, length, latitude, longitude, timestamp, image_path)
#         VALUES (?, ?, ?, ?, ?, ?)
#     ''', (species, length, latitude, longitude, timestamp, image_path))
#     conn.commit()
#     conn.close()

# def is_duplicate(species, length, latitude, longitude, timestamp_str):
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
#     c.execute("SELECT * FROM fish_records WHERE species = ? ORDER BY timestamp DESC", (species,))
#     records = c.fetchall()
#     conn.close()
#     print(records)
#     timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
#     msg = "No duplicate found in any condition"
#     duplicate_image_path = None

#     for record in records:
#         _, _, db_length, db_lat, db_lon, db_ts, db_image_path = record
#         db_time = datetime.strptime(db_ts, "%Y-%m-%d %H:%M:%S")

#         length_diff_ratio = abs(db_length - length) / db_length
#         geo_dist_km = geodesic((db_lat, db_lon), (latitude, longitude)).km
#         time_diff_sec = (timestamp - db_time).total_seconds()
#         days_diff = abs((timestamp - db_time).days)
        
#         print("check duplicate")
#         # Same coordinates (within 10m), time < 10 minutes, and length is close → Likely duplicate
#         if geo_dist_km < 0.01 and abs(time_diff_sec) < 600 and length_diff_ratio < 0.10:
#             msg = f"Duplicate detected — same location (<10m) {geo_dist_km*1000:.2f} meters, time <10min ({time_diff_sec}s), length match"
#             return True, msg, db_image_path
        
#         # Length difference too much → skip
#         print("length_diff_ratio", length_diff_ratio)
#         if length_diff_ratio > 0.10:
#             msg = f"Skipped due to length difference: {length_diff_ratio:.2%} (DB: {db_length}, Input: {length})"
#             continue

#         # Time > 1 day → always unique
#         print("days_diff", days_diff)
#         if days_diff >= 1:
#             msg = f"Unique due to time difference > 1 day: {days_diff} days"
#             continue

#         # Distance > 200m → skip
#         print(f"geo_dist_km {geo_dist_km*1000:.2f}")
#         if geo_dist_km > 0.2:
#             msg = f"Skipped due to distance > 200m: {geo_dist_km*1000:.2f} meters"
#             continue

#         # Distance < 200m, but time > 3 hours → likely unique
#         print(f"time_diff_sec {time_diff_sec/60:.2f} minutes")
        
#         if geo_dist_km <= 0.2 and time_diff_sec > 10800:
#             msg = f"Unique due to time > 3 hours at same location: {time_diff_sec/3600:.2f} hours"
#             continue

#     return False, msg, duplicate_image_path

# if __name__ == "__main__":
#     init_db()
#     init_db_for_data_collection()








import sqlite3
from datetime import datetime
from geopy.distance import geodesic
import logging
import os

# === DB files ===
DB_PATH = "D:\\Rahul Puri Data\\Projects\\Project GFL\\fish_records.db"
DB_PATH_DATA_COLLECTION = "D:\\Rahul Puri Data\\Projects\\Project GFL\\fish_data.db"


# =========================
# Schema + initialization
# =========================
def init_db():
    """Initialize primary DB (known/unknown fish + predicted)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Known species
    c.execute("""
        CREATE TABLE IF NOT EXISTS fish_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species TEXT,
            length REAL,
            latitude REAL,
            longitude REAL,
            timestamp TEXT,
            image_path TEXT
        )
    """)

    # Unknown species
    c.execute("""
        CREATE TABLE IF NOT EXISTS unknown_fish (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species TEXT,
            length REAL,
            latitude REAL,
            longitude REAL,
            timestamp TEXT,
            image_path TEXT
        )
    """)

    # # NEW: Predicted table (one row per API prediction)
    # c.execute("""
    #     CREATE TABLE IF NOT EXISTS predicted (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         species TEXT,                       -- species name used in response
    #         length_in REAL,                     -- measured length in inches
    #         uniqueness INTEGER,                 -- 0/1 (not is_dup)
    #         ai_uniqueness INTEGER,              -- 0/1 (not result["similar"])
    #         ai_uniqueness_distance REAL,        -- distance from compare_two_images
    #         image_path TEXT,                    -- filesystem path of annotated image
    #         image_url TEXT,                     -- full URL for sharing
    #         fish_confidence REAL,               -- optional: detector confidence
    #         species_confidence REAL,            -- optional: species detector confidence
    #         created_at TEXT DEFAULT (datetime('now'))  -- insertion time (UTC)
    #     )
    # """)

    # NEW: Predicted table (one row per API prediction)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predicted (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species TEXT,                       -- species name used in response
            length_in REAL,                     -- measured length in inches
            uniqueness INTEGER,                 -- 0/1 (not is_dup)
            ai_uniqueness INTEGER,              -- 0/1 (not result["similar"])
            ai_uniqueness_distance REAL,        -- distance from compare_two_images
            image_path TEXT,                    -- filesystem path of annotated image
            image_url TEXT,                     -- full URL for sharing
            server_time REAL,
            fish_confidence REAL,               -- optional: detector confidence
            created_at TEXT DEFAULT (datetime('now'))  -- insertion time (UTC)
        )
    """)


    # # Predicted
    # c.execute("""
    #     CREATE TABLE IF NOT EXISTS predicted (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         species TEXT,
    #         length_in REAL,
    #         uniqueness INTEGER,
    #         ai_uniqueness INTEGER,
    #         ai_uniqueness_distance REAL,
    #         image_path TEXT,
    #         image_url TEXT,
    #         server_time REAL,
    #         fish_confidence REAL,
    #         created_at TEXT DEFAULT (datetime('now'))
    #     )
    # """)

    # ✅ NEW: Models table
    c.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            species_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            json_file TEXT,  -- path or JSON string with metadata
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)


    # Helpful indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_fish_records_species ON fish_records(species)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_predicted_created_at ON predicted(created_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_models_project_id ON models(project_id)")

    conn.commit()
    conn.close()


def init_db_for_data_collection():
    """Initialize secondary DB for data collection (species/images)."""
    conn = sqlite3.connect(DB_PATH_DATA_COLLECTION)
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS species (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_name TEXT UNIQUE NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            handheld BOOLEAN NOT NULL,
            species_id INTEGER NOT NULL,
            FOREIGN KEY(species_id) REFERENCES species(id)
        )
    """)

    # ✅ Predefined species list
    species_list = [
        "Largemouth Bass: Micropterus nigricans",
        "Spotted Sea Trout: Cynoscion nebulosus",
        "Snook: Centropomus undecimalis",
        "Tarpon: Megalops atlanticus",
        "Mahi Mahi: Coryphaena hippurus",
        "Red Grouper: Epinephelus morio",
        "Black fin Tuna: Thunnus atlanticus",
        "Yellow Tail Snapper: Ocyurus chrysurus",
        "Jack Crevelle: Caranx hippos",
        "Red Drum: Sciaenops ocellatus"
    ]

    # Insert species (ignore duplicates if already exists)
    for sp in species_list:
        cursor.execute("""
            INSERT OR IGNORE INTO species (species_name) VALUES (?)
        """, (sp,))

    conn.commit()
    conn.close()
    print("✅ DB initialized with species list")


# =========================
# Insert helpers
# =========================
def insert_unknown_fish(species, length, latitude, longitude, timestamp, image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO unknown_fish (species, length, latitude, longitude, timestamp, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (species, length, latitude, longitude, timestamp, image_path))
    conn.commit()
    conn.close()


def insert_fish(species, length, latitude, longitude, timestamp, image_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO fish_records (species, length, latitude, longitude, timestamp, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (species, length, latitude, longitude, timestamp, image_path))
    conn.commit()
    conn.close()


def insert_predicted(
    species,
    length_in,
    uniqueness,
    ai_uniqueness,
    ai_uniqueness_distance,
    server_time,
    image_path,
    image_url,
    # base_url,
    fish_confidence=None,
    # species_confidence=None,
):
    """
    Store one prediction row. Booleans are stored as integers (0/1).
    """
    uniq_i = 1 if uniqueness else 0
    ai_uniq_i = 1 if ai_uniqueness else 0

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predicted (
            species, length_in, uniqueness, ai_uniqueness, ai_uniqueness_distance,
            image_path, image_url, server_time, fish_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        species, float(length_in), uniq_i, ai_uniq_i, float(ai_uniqueness_distance),
        float(server_time), 
        str(image_path), str(image_url), 
        # str(base_url),
        None if fish_confidence is None else float(fish_confidence),
        # None if species_confidence is None else float(species_confidence),
    ))
    conn.commit()
    conn.close()


# =========================
# Insert helpers
# =========================
def insert_model(project_id, species_name, model_path, json_file=None):
    """
    Insert a new model record into models table.
    - project_id: str (UUID or user-defined ID)
    - species_name: str (species handled by this model)
    - model_path: str (filesystem path to model file)
    - json_file: str (optional path to JSON metadata or JSON string)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO models (project_id, species_name, model_path, json_file)
        VALUES (?, ?, ?, ?)
    """, (project_id, species_name, model_path, json_file))
    conn.commit()
    conn.close()
    print(f"✅ Inserted model record for project {project_id}, species {species_name}")


def get_models_by_project(project_id):
    """Fetch all models linked to a given project_id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM models WHERE project_id = ?", (project_id,))
    rows = c.fetchall()
    conn.close()
    return rows



# =========================
# Duplicate check (existing)
# =========================
# Old work
def is_duplicate(species, length, latitude, longitude, timestamp_str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM fish_records WHERE species = ? ORDER BY timestamp DESC", (species,))
    records = c.fetchall()
    conn.close()

    print(records)
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    msg = "No duplicate found in any condition"
    duplicate_image_path = None

    for record in records:
        _, _, db_length, db_lat, db_lon, db_ts, db_image_path = record
        db_time = datetime.strptime(db_ts, "%Y-%m-%d %H:%M:%S")

        length_diff_ratio = abs(db_length - length) / db_length
        geo_dist_km = geodesic((db_lat, db_lon), (latitude, longitude)).km
        time_diff_sec = (timestamp - db_time).total_seconds()
        days_diff = abs((timestamp - db_time).days)

        print("check duplicate")

        # Same location (<10m), time <10min, length within 10% → duplicate
        if geo_dist_km < 0.01 and abs(time_diff_sec) < 600 and length_diff_ratio < 0.10:
            msg = (
                f"Duplicate detected — same location (<10m) {geo_dist_km*1000:.2f} meters, "
                f"time <10min ({time_diff_sec}s), length match"
            )
            return True, msg, db_image_path

        # Length too different → skip
        print("length_diff_ratio", length_diff_ratio)
        if length_diff_ratio > 0.10:
            msg = f"Skipped due to length difference: {length_diff_ratio:.2%} (DB: {db_length}, Input: {length})"
            continue

        # Time > 1 day → unique
        print("days_diff", days_diff)
        if days_diff >= 1:
            msg = f"Unique due to time difference > 1 day: {days_diff} days"
            continue

        # Distance > 200m → skip
        print(f"geo_dist_km {geo_dist_km*1000:.2f}")
        if geo_dist_km > 0.2:
            msg = f"Skipped due to distance > 200m: {geo_dist_km*1000:.2f} meters"
            continue

        # Distance <= 200m and time > 3h → likely unique
        print(f"time_diff_sec {time_diff_sec/60:.2f} minutes")
        if geo_dist_km <= 0.2 and time_diff_sec > 10800:
            msg = f"Unique due to time > 3 hours at same location: {time_diff_sec/3600:.2f} hours"
            continue

    return False, msg, duplicate_image_path


#new duplicate logic only time stamp
def is_duplicate_v2(timestamp_str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # ✅ Fetch all records regardless of species
    c.execute("SELECT timestamp, image_path FROM fish_records ORDER BY timestamp DESC")
    records = c.fetchall()
    conn.close()
    # print("MY RECORDS: ", records)
 
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    msg = "No duplicate found in 24h window"
    duplicate_image_path = None
 
    for db_ts, db_image_path in records:
        db_time = datetime.strptime(db_ts, "%Y-%m-%d %H:%M:%S")
 
        # ✅ Check only 24h window
        if abs((timestamp - db_time).total_seconds()) <= 86400:
            msg = f"Potential duplicate within 24h (DB time: {db_time}, Input: {timestamp})"
            return True, msg, db_image_path
 
    return False, msg, duplicate_image_path
 


if __name__ == "__main__":
    init_db()
    init_db_for_data_collection()

    # # Example insert
    # insert_model(
    #     project_id="ab07c1f5-529f-4feb-8183-276c4b880011",
    #     species_name="Red Drum (Sciaenops ocellatus)",
    #     model_path="D:/Rahul Puri Data/Projects/Project GFL/app_backend/modules/fish_training/models/model_group_xxx.pt",
    #     json_file="D:/Rahul Puri Data/Projects/Project GFL/app_backend/modules/fish_training/metadata/group_hashes_current.json"
    # )

    # print(get_models_by_project("ab07c1f5-529f-4feb-8183-276c4b880011"))















