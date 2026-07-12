


import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

def read_species_map(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    name_col: str
) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
    cur = conn.cursor()
    cur.execute(f"SELECT {id_col}, {name_col} FROM {table}")
    rows = [(int(sid), str(sname).strip()) for (sid, sname) in cur.fetchall()]
    if not rows:
        raise RuntimeError(f"No rows in species map table '{table}'.")
    rows.sort(key=lambda r: r[0])
    id_to_name = {sid: name for sid, name in rows}
    return id_to_name, rows


def fetch_records(
    conn: sqlite3.Connection,
    table: str,
    image_col: str,
    species_col: str,
    handheld_col: str,
    id_col: str | None = None
) -> List[Tuple[str, int, bool, str]]:
    """
    returns (name_id, species_id, handheld_bool, image_path_or_url)
    """
    cur = conn.cursor()
    cols: List[str] = []
    if id_col:
        cols.append(id_col)
    cols += [image_col, handheld_col, species_col]
    cur.execute(f"SELECT {', '.join(cols)} FROM {table}")
    rows = cur.fetchall()

    out: List[Tuple[str, int, bool, str]] = []
    for row in rows:
        if id_col:
            rec_id, img_path, handheld, species_id = row
            name_id = str(rec_id)
        else:
            img_path, handheld, species_id = row
            name_id = Path(str(img_path)).stem
        out.append((name_id, int(species_id), bool(handheld), str(img_path)))
    return out



























# import sqlite3
# from pathlib import Path
# from typing import Dict, List, Tuple

# def read_species_map(
#     conn: sqlite3.Connection,
#     table: str,
#     id_col: str,
#     name_col: str
# ) -> Tuple[Dict[int, str], List[Tuple[int, str]]]:
#     cur = conn.cursor()
#     cur.execute(f"SELECT {id_col}, {name_col} FROM {table}")
#     rows = [(int(sid), str(sname).strip()) for (sid, sname) in cur.fetchall()]
#     if not rows:
#         raise RuntimeError(f"No rows in species map table '{table}'.")
#     rows.sort(key=lambda r: r[0])
#     id_to_name = {sid: name for sid, name in rows}
#     return id_to_name, rows


# def fetch_records(
#     conn: sqlite3.Connection,
#     table: str,
#     image_col: str,
#     species_col: str,
#     handheld_col: str,
#     id_col: str | None = None
# ) -> List[Tuple[str, int, bool, str]]:
#     """
#     returns (name_id, species_id, handheld_bool, image_path)
#     """
#     cur = conn.cursor()
#     cols: List[str] = []
#     if id_col:
#         cols.append(id_col)
#     cols += [image_col, handheld_col, species_col]
#     cur.execute(f"SELECT {', '.join(cols)} FROM {table}")
#     rows = cur.fetchall()

#     out: List[Tuple[str, int, bool, str]] = []
#     for row in rows:
#         if id_col:
#             rec_id, img_path, handheld, species_id = row
#             name_id = str(rec_id)
#         else:
#             img_path, handheld, species_id = row
#             name_id = Path(str(img_path)).stem
#         out.append((name_id, int(species_id), bool(handheld), str(img_path)))
#     return out
