# Install the Unicode-aware regex package and upload exactly ten CSV files.
import io
import json
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "regex"],
    check=True,
)

import pandas as pd
import regex
from google.colab import files

input_files = []
NUMBER_OF_FILES = 10
for upload_number in range(1, NUMBER_OF_FILES + 1):
    print(f"Upload CSV file {upload_number} of {NUMBER_OF_FILES}:")
    one_upload = files.upload()
    uploaded_csvs = [
        (name, data) for name, data in one_upload.items()
        if name.lower().endswith(".csv")
    ]
    if len(uploaded_csvs) != 1:
        raise ValueError(
            f"Upload exactly one CSV in upload window {upload_number}; "
            f"received {[name for name, _ in uploaded_csvs]}"
        )
    filename, file_bytes = uploaded_csvs[0]
    input_files.append((upload_number, filename, file_bytes))

OUTPUT_DIR = Path("/content/weibo_10_candidate_files_merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Input CSV files:", [name for _, name, _ in input_files])



# Read and merge all ten files while preserving source file and upload order.
frames = []
for upload_number, filename, file_bytes in input_files:
    frame = pd.read_csv(
        io.BytesIO(file_bytes),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    frame["source_file"] = filename
    frame["upload_number"] = upload_number
    frame["source_row"] = range(2, len(frame) + 2)  # CSV row number including header
    frames.append(frame)
    print(filename, "rows:", len(frame))

merged = pd.concat(frames, ignore_index=True, sort=False).fillna("")
required = {"微博正文", "发布时间"}
missing = required - set(merged.columns)
if missing:
    raise ValueError(f"Required columns are missing: {sorted(missing)}")

MERGED_FILE = OUTPUT_DIR / "eligible_candidates_merged_before_cleaning.csv"
merged.to_csv(MERGED_FILE, index=False, encoding="utf-8-sig")
print("Merged rows:", len(merged))
print("Merged backup:", MERGED_FILE)
