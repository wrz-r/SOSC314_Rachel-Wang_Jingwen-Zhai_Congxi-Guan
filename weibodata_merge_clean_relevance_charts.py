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

