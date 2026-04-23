from pathlib import Path


  # <-- change thisfrom pathlib import Path
import re
import pandas as pd
# --- CONFIG ---
INPUT_DIR = Path(r"D:\University FIU\CAPSTONE\annotations_templates") 
OUTPUT_FILE = INPUT_DIR / "master_roles.csv"
PATTERN = "Novel*_roles.csv"

EXPECTED_COLS = [
    "doc_id", "chain_id", "animacy", "character", "canonical_name",
    "mentions", "role", "confidence", "notes"
]

ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
CHUNKSIZE = 100_000

files = sorted(INPUT_DIR.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"No files matched {PATTERN} in {INPUT_DIR}")

first_write = True

for f in files:
    print(f"Reading: {f.name}")

    m = re.match(r"(.+)_roles$", f.stem)
    novel = m.group(1) if m else f.stem

    read_ok = False

    # Try encodings strictly first
    for enc in ENCODINGS_TO_TRY:
        try:
            for chunk in pd.read_csv(
                f,
                chunksize=CHUNKSIZE,
                encoding=enc,
                encoding_errors="strict",   # fail fast if wrong encoding
                dtype=str,                  # keeps things consistent
            ):
                missing = [c for c in EXPECTED_COLS if c not in chunk.columns]
                if missing:
                    raise ValueError(f"{f.name} missing columns: {missing}\nFound: {list(chunk.columns)}")

                chunk.insert(0, "novel", novel)
                chunk.insert(1, "source_file", f.name)
                chunk = chunk[["novel", "source_file"] + EXPECTED_COLS]

                chunk.to_csv(
                    OUTPUT_FILE,
                    index=False,
                    mode="w" if first_write else "a",
                    header=first_write,
                )
                first_write = False

            print(f"  ✅ OK with encoding={enc}")
            read_ok = True
            break

        except UnicodeDecodeError:
            # try next encoding
            continue

    if read_ok:
        continue

    # Last resort: replace undecodable bytes so merge can finish
    print(f"  ⚠️ Could not decode {f.name} with {ENCODINGS_TO_TRY}. Using replacement mode.")
    for chunk in pd.read_csv(
        f,
        chunksize=CHUNKSIZE,
        encoding="utf-8",
        encoding_errors="replace",  # replace bad bytes with �
        engine="python",            # sometimes more forgiving
        dtype=str,
        on_bad_lines="skip",        # if a row is structurally broken
    ):
        missing = [c for c in EXPECTED_COLS if c not in chunk.columns]
        if missing:
            raise ValueError(f"{f.name} missing columns: {missing}\nFound: {list(chunk.columns)}")

        chunk.insert(0, "novel", novel)
        chunk.insert(1, "source_file", f.name)
        chunk = chunk[["novel", "source_file"] + EXPECTED_COLS]

        chunk.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )
        first_write = False

print(f"\n✅ Done. Master saved to: {OUTPUT_FILE}")
