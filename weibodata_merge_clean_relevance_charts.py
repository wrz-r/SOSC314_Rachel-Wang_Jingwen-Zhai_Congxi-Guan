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



# Reapply the same content-cleaning rules used in the latest crawler notebook.
FICTION_TERMS = ["小说", "阅读全文", "完结", "大结局", "晋江文学", "txt", "全章节"]
EXCLUDE_TERMS = ["微博问答", "超话", "恋与深空", "恋与制作人", "微博正文"]
CELEBRITY_TERMS = [
    "谢娜", "何炅", "杨幂", "Angelababy", "陈坤", "赵丽颖", "易烊千玺", "王源", "王俊凯",
    "姚晨", "张杰", "迪丽热巴", "唐嫣", "林心如", "邓超", "陈乔恩", "刘亦菲", "杨紫",
    "宋茜", "赵薇", "郭德纲", "林志颖", "胡歌", "范冰冰", "王力宏", "陈赫", "黄子韬",
    "鹿晗", "贾乃亮", "罗志祥", "黄晓明", "薛之谦", "杨洋", "李晨", "林俊杰", "韩庚",
    "林更新", "刘烨", "张艺兴", "高圆圆", "刘涛", "孙俪", "王珞丹", "佟丽娅", "关晓彤",
    "吴磊", "戚薇", "范玮琪", "郑恺", "王祖蓝", "马伊琍", "张靓颖", "蔡依林", "赵露思",
    "郭采洁", "周冬雨", "虞书欣", "章子怡", "白鹿", "李沁", "景甜", "倪妮", "张馨予",
    "白百何", "古力娜扎", "江疏影", "杨超越", "肖战", "王一博", "马嘉祺", "宋亚轩",
    "刘耀文", "丁程鑫", "严浩翔", "张真源", "贺峻霖",
]

fiction_pattern = regex.compile(
    "|".join(map(regex.escape, FICTION_TERMS)), regex.IGNORECASE
)
exclude_pattern = regex.compile("|".join(map(regex.escape, EXCLUDE_TERMS)))
celebrity_pattern = regex.compile(
    "|".join(map(regex.escape, CELEBRITY_TERMS)), regex.IGNORECASE
)
other_script_pattern = regex.compile(
    r"(?!\p{Script=Han}|\p{Script=Latin})\p{L}"
)
han_character_pattern = regex.compile(r"\p{Script=Han}")

def find_cleaning_reasons(row):
    body = str(row.get("微博正文", "")).strip()
    reasons = []
    if not body:
        reasons.append("blank_body")

    query_keyword = str(row.get("query_keyword", "")).strip()
    if query_keyword and query_keyword not in body:
        reasons.append("query_keyword_not_in_body:" + query_keyword)

    fiction = fiction_pattern.search(body)
    excluded = exclude_pattern.search(body)
    celebrity = celebrity_pattern.search(body)
    other_script = other_script_pattern.search(body)
    if fiction:
        reasons.append("fiction_term:" + fiction.group())
    if excluded:
        reasons.append("excluded_term:" + excluded.group())
    if celebrity:
        reasons.append("celebrity_name:" + celebrity.group())
    if other_script:
        reasons.append("other_script:" + other_script.group())
    if body and not han_character_pattern.search(body):
        reasons.append("no_han_characters")
    return "; ".join(reasons)

merged["cleaning_removal_reason"] = merged.apply(find_cleaning_reasons, axis=1)
cleaning_removed = merged[merged["cleaning_removal_reason"] != ""].copy()
cleaned = merged[merged["cleaning_removal_reason"] == ""].copy()

CLEANING_REMOVED_FILE = OUTPUT_DIR / "removed_by_cleaning.csv"
CLEANED_BEFORE_RELEVANCE_FILE = OUTPUT_DIR / "cleaned_before_relevance_filter.csv"
cleaning_removed.to_csv(CLEANING_REMOVED_FILE, index=False, encoding="utf-8-sig")
cleaned.to_csv(CLEANED_BEFORE_RELEVANCE_FILE, index=False, encoding="utf-8-sig")

print("Rows before cleaning:", len(merged))
print("Rows removed by cleaning:", len(cleaning_removed))
print("Rows retained before relevance filtering:", len(cleaned))
