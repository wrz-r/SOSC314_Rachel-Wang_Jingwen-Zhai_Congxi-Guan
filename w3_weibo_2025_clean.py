# this is a new colab code.
# 1. Upload the THREE ZIP archives using Colab's left sidebar (Files > Upload).
# Each ZIP may contain its own weibo-search-2025-full/结果文件/... directory.
from pathlib import Path
from zipfile import ZipFile
import csv

DATA_DIR = Path("/content")
ZIP_FILES = sorted(DATA_DIR.glob("*.zip"))
if len(ZIP_FILES) != 3:
    raise FileNotFoundError(
        f"Expected 3 uploaded ZIP files in {DATA_DIR}; found {len(ZIP_FILES)}: "
        f"{[p.name for p in ZIP_FILES]}"
    )
MONTHS = {"2025-01", "2025-02", "2025-04", "2025-05", "2025-10", "2025-11"}
COLUMNS = ["id", "user_id", "微博正文", "发布时间", "关键词"]
for path in ZIP_FILES:
    print("Ready:", path.name, f"({path.stat().st_size / 1024**2:.1f} MB)")


# 2. Read keyword CSVs from each ZIP; merge only the six selected months.
# 关键词 records the keyword folder/search query that produced this source row.
from collections import Counter
from pathlib import PurePosixPath
from io import TextIOWrapper

MERGED = DATA_DIR / "weibo_2025_6months_merged.csv"
month_counts = Counter()
files_read = 0
with MERGED.open("w", encoding="utf-8-sig", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=COLUMNS)
    writer.writeheader()
    for zip_path in ZIP_FILES:
        with ZipFile(zip_path) as archive:
            for member in archive.infolist():
                parts = PurePosixPath(member.filename).parts
                if member.is_dir() or "结果文件" not in parts or not member.filename.endswith(".csv"):
                    continue
                pos = parts.index("结果文件")
                if len(parts) != pos + 3:
                    continue
                keyword = parts[pos + 1]
                if parts[-1] != f"{keyword}.csv":
                    continue  # Ignore merged CSVs, logs, and macOS metadata files.
                with archive.open(member) as binary, TextIOWrapper(
                    binary, encoding="utf-8-sig", newline=""
                ) as text:
                    reader = csv.DictReader(text)
                    required = set(COLUMNS) - {"关键词"}
                    if not reader.fieldnames or not required.issubset(reader.fieldnames):
                        raise ValueError(f"Unexpected CSV columns in {zip_path.name}: {member.filename}")
                    files_read += 1
                    for row in reader:
                        month = (row.get("发布时间") or "")[:7]
                        if month not in MONTHS:
                            continue
                        writer.writerow({
                            "id": (row.get("id") or "").lstrip("\ufeff"),
                            "user_id": row.get("user_id") or "",
                            "微博正文": row.get("微博正文") or "",
                            "发布时间": row.get("发布时间") or "",
                            "关键词": keyword,
                        })
                        month_counts[month] += 1

if not sum(month_counts.values()):
    raise ValueError("No target-month posts were found. Check the uploaded ZIP files.")
print("Keyword CSV files read:", files_read, "| Rows merged:", sum(month_counts.values()))
print("Rows by month:", dict(sorted(month_counts.items())))
if MONTHS - month_counts.keys():
    print("No rows found for:", sorted(MONTHS - month_counts.keys()))
print("Merged CSV:", MERGED)
