"""Cross-check the shipped class_map against the model's embedded metadata, and
emit a canonical index-aligned labels.txt.

Why: Names_Model_*.txt is NOT a plain label list. It is 340 lines --
a `class_map:` header followed by `  <idx>: <Common Name> (<Binomial>)`.
A recipient doing the obvious `readLines()` gets index 0 = "class_map:" and every
species shifted by one. TFLITE_CONTRACT.md v1 called it "339 entries", which
actively invites that bug.

The authoritative order lives inside the ORIGINAL .tflite: tflite_support appends
a zip containing temp_meta.txt, a Python-repr dict with a `names` mapping. We
compare the two and emit labels.txt (339 lines, one name per line, line i == class i).

Usage:
  python export/make_labels.py --class-map <Names_*.txt> \
      --metadata-model <original_float32.tflite> --out <labels.txt>
"""

import argparse
import ast
import io
import re
import zipfile
from pathlib import Path


def parse_class_map(p: Path):
    """`class_map:` header + '  <idx>: <name>' lines -> {idx: name}."""
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(\d+):\s*(.+?)\s*$", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def parse_embedded_names(tflite: Path):
    """Pull the `names` dict out of the metadata zip appended to the .tflite."""
    b = tflite.read_bytes()
    try:
        z = zipfile.ZipFile(io.BytesIO(b))
    except zipfile.BadZipFile:
        return None
    meta = ast.literal_eval(z.read(z.namelist()[0]).decode())
    return meta.get("names")


def norm(s: str) -> str:
    """'Black fin Tuna (Thunnus atlanticus)' -> 'blackfintunathunnusatlanticus'."""
    return re.sub(r"[^a-z]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-map", required=True, type=Path)
    ap.add_argument("--metadata-model", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cm = parse_class_map(args.class_map)
    n_lines = len(args.class_map.read_text(encoding="utf-8").splitlines())
    print(f"class_map file: {n_lines} lines -> {len(cm)} parsed entries")
    print(f"  (naive readLines() would give {n_lines} 'labels' and shift every class by one)")

    idxs = sorted(cm)
    assert idxs == list(range(len(idxs))), "class_map indices are not contiguous from 0"
    print(f"  indices contiguous 0..{idxs[-1]}")

    if args.metadata_model and args.metadata_model.exists():
        emb = parse_embedded_names(args.metadata_model)
        if emb is None:
            print("  metadata model carries no embedded names zip -- skipped cross-check")
        else:
            print(f"embedded metadata names: {len(emb)} entries")
            assert len(emb) == len(cm), f"count mismatch: embedded {len(emb)} vs class_map {len(cm)}"
            bad = [i for i in idxs if norm(emb[i]) != norm(cm[i])]
            if bad:
                print(f"  !! {len(bad)} MISMATCHES, first 5:")
                for i in bad[:5]:
                    print(f"     [{i}] embedded={emb[i]!r}  class_map={cm[i]!r}")
                raise SystemExit("class_map and embedded metadata disagree -- do not ship")
            print(f"  cross-check PASSED: all {len(cm)} names agree with the model's own metadata")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(cm[i] for i in idxs) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}: {len(idxs)} lines, line i == class i")
    print(f"  [0]   {cm[0]}")
    print(f"  [338] {cm[338]}")


if __name__ == "__main__":
    main()
