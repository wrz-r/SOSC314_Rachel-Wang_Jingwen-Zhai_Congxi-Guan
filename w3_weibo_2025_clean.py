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


# 3. Deduplicate globally by the first 20 characters of 微博正文.
# Retain the earliest published row, but combine all original search keywords.
import sqlite3
import tempfile

DEDUPED = DATA_DIR / "weibo_2025_6months_prefix20.csv"
with tempfile.TemporaryDirectory(dir=DATA_DIR) as staging:
    db = sqlite3.connect(str(Path(staging) / "dedup.sqlite"))
    db.execute("""CREATE TABLE posts (
        prefix TEXT PRIMARY KEY, id TEXT, user_id TEXT,
        body TEXT, published_at TEXT
    )""")
    db.execute("""CREATE TABLE keywords (
        prefix TEXT, keyword TEXT, PRIMARY KEY (prefix, keyword)
    )""")
    with MERGED.open(encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.DictReader(source), start=1):
            body = row["微博正文"]
            # Keep blank bodies distinct so the cleaning step can remove/report each one.
            prefix = body[:20] if body else f"\0EMPTY:{row_number}"
            db.execute("""INSERT INTO posts VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(prefix) DO UPDATE SET
                    id=excluded.id, user_id=excluded.user_id,
                    body=excluded.body, published_at=excluded.published_at
                WHERE excluded.published_at < posts.published_at
            """, (prefix, row["id"], row["user_id"], body, row["发布时间"]))
            db.execute("INSERT OR IGNORE INTO keywords VALUES (?, ?)",
                       (prefix, row["关键词"]))
            if row_number % 10000 == 0:
                db.commit()
    db.commit()

    with DEDUPED.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(COLUMNS)
        count = 0
        query = """SELECT id, user_id, body, published_at,
          (SELECT group_concat(keyword, '、') FROM (
              SELECT keyword FROM keywords WHERE prefix=posts.prefix ORDER BY keyword
          )) AS source_keywords
          FROM posts ORDER BY published_at, id"""
        for row in db.execute(query):
            writer.writerow(row)
            count += 1
    db.close()

print("Rows after first-20-character deduplication:", count)
print("Deduplicated CSV:", DEDUPED)



# 4. Remove the requested terms and non-Han/non-Latin letters.
# Save all removed rows with reasons; leave the merged/deduplicated CSVs untouched.
import subprocess
import sys
try:
    import regex
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "regex"], check=True)
    import importlib
    import site
    site.addsitedir(site.getusersitepackages())
    importlib.invalidate_caches()
    import regex

FICTION_TERMS = ["小说", "阅读全文", "完结", "大结局", "晋江文学", "txt", "全章节"]
EXCLUDE_TERMS = ["微博问答", "超话", "恋与深空", "恋与制作人", "微博正文"]
fiction_pattern = regex.compile("|".join(map(regex.escape, FICTION_TERMS)), regex.IGNORECASE)
exclude_pattern = regex.compile("|".join(map(regex.escape, EXCLUDE_TERMS)))
other_script = regex.compile(r"(?!\p{Script=Han}|\p{Script=Latin})\p{L}")

CLEANED = DATA_DIR / "weibo_2025_6months_cleaned.csv"
REMOVED = DATA_DIR / "weibo_2025_6months_removed.csv"
counts = Counter()
with (DEDUPED.open(encoding="utf-8-sig", newline="") as source,
      CLEANED.open("w", encoding="utf-8-sig", newline="") as clean_out,
      REMOVED.open("w", encoding="utf-8-sig", newline="") as removed_out):
    reader = csv.DictReader(source)
    kept_writer = csv.DictWriter(clean_out, fieldnames=COLUMNS)
    removed_writer = csv.DictWriter(removed_out, fieldnames=COLUMNS + ["removal_reason"])
    kept_writer.writeheader()
    removed_writer.writeheader()
    for row in reader:
        body = row["微博正文"] or ""
        reasons = []
        if not body.strip():
            reasons.append("empty_body")
        fiction = fiction_pattern.search(body)
        excluded = exclude_pattern.search(body)
        foreign = other_script.search(body)
        if fiction:
            reasons.append(f"fiction_term:{fiction.group()}")
        if excluded:
            reasons.append(f"excluded_term:{excluded.group()}")
        if foreign:
            reasons.append(f"other_script:{foreign.group()}")
        if reasons:
            removed_writer.writerow({**row, "removal_reason": "; ".join(reasons)})
            counts["removed"] += 1
            for reason in reasons:
                counts[reason.split(":")[0]] += 1
        else:
            kept_writer.writerow(row)
            counts["kept"] += 1

print("Kept:", counts["kept"], "| Removed:", counts["removed"])
print("Rule hits (a post may hit multiple):",
      {name: counts[name] for name in ("empty_body", "fiction_term", "excluded_term", "other_script")})
print("Cleaned CSV:", CLEANED)
print("Removed rows and reasons:", REMOVED)



# 5. Plot cleaned post counts by month and by search keyword.
# Run AFTER cleaning (cell 4). Figures and summary CSVs are saved in DATA_DIR.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter

MONTH_ORDER = ["2025-01", "2025-02", "2025-04", "2025-05", "2025-10", "2025-11"]
ENGLISH = {
    "婚姻": "Marriage", "结婚": "Getting married", "婚恋": "Marriage & dating",
    "恋爱": "Dating", "对象": "Partner", "伴侣": "Companion",
    "相亲": "Matchmaking", "单身": "Single", "不婚": "Not marrying",
    "晚婚": "Late marriage", "恐婚": "Fear of marriage", "催婚": "Marriage pressure",
    "婚姻登记": "Marriage registration", "彩礼": "Bride price",
    "离婚冷静期": "Divorce cooling-off period",
}

# Streaming read limits memory use for large cleaned CSV files.
if not CLEANED.exists():
    raise FileNotFoundError(f"Run the cleaning cell first: {CLEANED}")
month_counts = Counter()
keyword_counts = Counter()
unique_posts = 0
for chunk in pd.read_csv(CLEANED, dtype=str, chunksize=50000, encoding="utf-8-sig"):
    if "发布时间" not in chunk or "关键词" not in chunk:
        raise ValueError("Cleaned CSV must contain 发布时间 and 关键词 columns.")
    unique_posts += len(chunk)
    months = chunk["发布时间"].fillna("").str.slice(0, 7)
    month_counts.update(months[months.isin(MONTH_ORDER)].value_counts().to_dict())
    words = chunk["关键词"].fillna("").str.split("、").explode().str.strip()
    keyword_counts.update(words[words.ne("")].value_counts().to_dict())
if not unique_posts:
    raise ValueError("The cleaned CSV has no posts to plot.")

# Save exact numeric summaries alongside the plots.
monthly = pd.DataFrame({"month": MONTH_ORDER,
                        "cleaned_posts": [month_counts[m] for m in MONTH_ORDER]})
all_keywords = list(ENGLISH) + sorted(set(keyword_counts) - set(ENGLISH))
by_keyword = pd.DataFrame([(word, keyword_counts[word]) for word in all_keywords],
                          columns=["keyword", "cleaned_posts"])
by_keyword = by_keyword.sort_values(["cleaned_posts", "keyword"], ascending=[False, True])
monthly.to_csv(DATA_DIR / "weibo_2025_cleaned_monthly_counts.csv", index=False, encoding="utf-8-sig")
by_keyword.to_csv(DATA_DIR / "weibo_2025_cleaned_keyword_counts.csv", index=False, encoding="utf-8-sig")


# Chart 1: discrete month totals. Other months were not collected here.
fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.bar([m[5:] for m in MONTH_ORDER], monthly["cleaned_posts"], color="#3971aa", width=0.67)
ax.bar_label(bars, padding=3, fmt="%d")
ax.set(title="Cleaned Weibo posts by month (2025)", xlabel="Month", ylabel="Number of posts")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(0, max(monthly["cleaned_posts"]) * 1.17 + 1)
fig.text(0.02, 0.01,
         "Note: Only Jan, Feb, Apr, May, Oct, and Nov were imported.",
         fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, 0.09, 1, 1))
MONTH_PNG = DATA_DIR / "weibo_2025_cleaned_monthly_counts.png"
fig.savefig(MONTH_PNG, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)


# Chart 2: original search keywords. English labels render reliably in Colab.
labels = [ENGLISH.get(k, k) for k in by_keyword["keyword"]]
fig, ax = plt.subplots(figsize=(10, max(5, 0.43 * len(labels) + 1.4)))
bars = ax.barh(labels, by_keyword["cleaned_posts"], color="#67a19a")
ax.invert_yaxis()
ax.bar_label(bars, padding=4, fmt="%d")
ax.set(title="Cleaned Weibo posts by search keyword (2025)", xlabel="Number of posts")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(0, max(by_keyword["cleaned_posts"]) * 1.18 + 1)
fig.tight_layout(rect=(0, 0.075, 1, 1))
KEYWORD_PNG = DATA_DIR / "weibo_2025_cleaned_keyword_counts.png"
fig.savefig(KEYWORD_PNG, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Unique cleaned posts: {unique_posts:,}")
print("Saved charts:", MONTH_PNG, "and", KEYWORD_PNG)
print("Saved count tables in:", DATA_DIR)
