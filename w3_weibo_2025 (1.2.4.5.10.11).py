# 1. Use the same local Colab repository location as in the working version.
from pathlib import Path
import subprocess
import sys

FULL_REPO = Path("/content/weibo-search-2025-full")
if not FULL_REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/dataabc/weibo-search.git", str(FULL_REPO)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(FULL_REPO / "requirements.txt")], check=True)
# Keep the versions used in your earlier working Colab notebook.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                "Scrapy==2.12.0", "Twisted==24.11.0"], check=True)
print("Repository:", FULL_REPO)


# 2. Enter your Weibo cookie and use only the group's marriage keywords.
from getpass import getpass

WEIBO_COOKIE = getpass("Paste your Weibo cookie here: ").strip()
if not WEIBO_COOKIE:
    raise ValueError("A Weibo cookie is required.")

KEYWORD_GROUPS = {
    "marriage": [
        "婚姻", "结婚", "婚恋", "恋爱", "对象", "伴侣", "相亲", "单身",
        "不婚", "晚婚", "恐婚", "催婚", "婚姻登记", "彩礼", "离婚冷静期",
    ],
}
KEYWORDS = KEYWORD_GROUPS["marriage"]
DOWNLOAD_DELAY = 15
ALLOW_EMPTY_RESULTS = False  # Pause if a whole month yields zero new rows.
print("Cookie loaded (hidden). Keywords:", len(KEYWORDS), "| monthly crawls: 12")



# 3. Set the year and the first/last months to collect (inclusive).
import pandas as pd

YEAR = 2025
START_MONTH = 1
END_MONTH = 3
# set the month to what we want (4-6, 10-12, etc)
assert 1 <= START_MONTH <= END_MONTH <= 12

first_month = pd.Timestamp(YEAR, START_MONTH, 1)
following_month = pd.Timestamp(YEAR, END_MONTH, 1) + pd.offsets.MonthBegin(1)
month_starts = pd.date_range(first_month, following_month, freq="MS")
month_ranges = [
    (start.strftime("%Y-%m"), start.strftime("%Y-%m-%d"),
     (following - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    for start, following in zip(month_starts[:-1], month_starts[1:])
]
print("First:", month_ranges[0], "Last:", month_ranges[-1])



# 4. Run ONE crawler process per month, searching all 15 keywords in that process.
# The crawler still queries each keyword internally. Zero-result months stop for review.
import csv
import json
import os
import time
from collections import deque

LOG_FILE = Path("/content/weibo_2025_monthly_log.csv")
DEBUG_DIR = Path("/content/weibo_2025_debug_logs")
DEBUG_DIR.mkdir(exist_ok=True)
FIELDS = ["month", "start_date", "end_date", "new_rows", "return_code",
          "elapsed_seconds", "status", "sizes_before"]

history = {}
if LOG_FILE.exists():
    with LOG_FILE.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            history[row["month"]] = row

def save_record(row):
    new_log = not LOG_FILE.exists()
    with LOG_FILE.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_log:
            writer.writeheader()
        writer.writerow(row)
    history[row["month"]] = row

def count_rows(path):
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)

for month, start_date, end_date in month_ranges:
    previous = history.get(month)
    if previous and previous["status"] == "done":
        print("Already completed:", month)
        continue

    files = {keyword: FULL_REPO / "结果文件" / keyword / f"{keyword}.csv"
             for keyword in KEYWORDS}
    # A retry must first remove every partial append from an interrupted month.
    if previous and previous["status"] != "done":
        for keyword, size in json.loads(previous["sizes_before"]).items():
            path = files[keyword]
            if path.exists():
                if path.stat().st_size < size:
                    raise RuntimeError(f"Output CSV became shorter unexpectedly: {path}")
                if size == 0:
                    path.unlink()  # CsvPipeline writes its header only for a new file.
                else:
                    with path.open("r+b") as f:
                        f.truncate(size)

    rows_before = {keyword: count_rows(path) for keyword, path in files.items()}
    sizes_before = {keyword: path.stat().st_size if path.exists() else 0
                    for keyword, path in files.items()}
    environment = os.environ.copy()
    environment.update({
        "WEIBO_COOKIE": WEIBO_COOKIE,
        "WEIBO_KEYWORDS": json.dumps(KEYWORDS, ensure_ascii=False),
        "WEIBO_START_DATE": start_date,
        "WEIBO_END_DATE": end_date,
        "WEIBO_TYPE": "1",          # Only original Weibo posts
        "WEIBO_CONTAIN_TYPE": "0",  # No media-type restriction
        "WEIBO_REGION": json.dumps(["全部"], ensure_ascii=False),
        "WEIBO_LIMIT_RESULT": "0",  # No numerical crawler cap
        "WEIBO_FETCH_IP": "0",
    })
    record = dict(month=month, start_date=start_date, end_date=end_date,
                  new_rows=0, return_code="", elapsed_seconds=0,
                  status="started", sizes_before=json.dumps(sizes_before, ensure_ascii=False))
    save_record(record)
    log_path = DEBUG_DIR / f"{month}_all_marriage_keywords.log"
    print(f"\nCollecting: {month} | all {len(KEYWORDS)} keywords", flush=True)
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_output:
        result = subprocess.run(
            ["scrapy", "crawl", "search", "-s", "LOG_LEVEL=INFO",
             "-s", f"DOWNLOAD_DELAY={DOWNLOAD_DELAY}",
             "-s", "RANDOMIZE_DOWNLOAD_DELAY=False",
             "-s", "CONCURRENT_REQUESTS=1"],
            cwd=FULL_REPO, env=environment, stdout=log_output,
            stderr=subprocess.STDOUT, text=True,
        )
    new_rows = sum(count_rows(path) - rows_before[keyword]
                   for keyword, path in files.items())
    record.update(new_rows=new_rows, return_code=result.returncode,
                  elapsed_seconds=round(time.monotonic() - t0, 1),
                  status="done" if result.returncode == 0 and
                  (new_rows > 0 or ALLOW_EMPTY_RESULTS) else "failed")
    save_record(record)
    print(f"New rows across all keywords: {new_rows} | "
          f"Seconds: {record['elapsed_seconds']} | Return code: {result.returncode}")
    if record["status"] == "failed":
        with log_path.open(encoding="utf-8", errors="replace") as f:
            print("Crawler diagnostics:\n", "".join(deque(f, maxlen=45)))
        print("Full diagnostics:", log_path)
        raise RuntimeError(f"Check the results for {month} before retrying. "
                           "Set ALLOW_EMPTY_RESULTS=True only if zero rows are expected.")

print(f"All {len(month_ranges)} selected monthly searches completed. Run the next cell to combine results.")



