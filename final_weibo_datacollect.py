# Install the tested crawler version and Python dependencies.
from pathlib import Path
import subprocess
import sys

FULL_REPO = Path("/content/weibo-search-sampling")
PINNED_COMMIT = "b4535b71d36ae61d13ab0083a18a152914f14ba7"

if not FULL_REPO.exists():
    subprocess.run(
        ["git", "clone", "--no-checkout",
         "https://github.com/dataabc/weibo-search.git", str(FULL_REPO)],
        check=True,
    )
subprocess.run(
    ["git", "-C", str(FULL_REPO), "fetch", "--depth", "1", "origin", PINNED_COMMIT],
    check=True,
)
subprocess.run(
    ["git", "-C", str(FULL_REPO), "checkout", "--detach", PINNED_COMMIT],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(FULL_REPO / "requirements.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
     "Scrapy==2.12.0", "Twisted==24.11.0", "regex"],
    check=True,
)
print("Repository ready:", FULL_REPO)


# EDIT THE YEAR HERE.
from getpass import getpass
from datetime import date
# the year and date can be changed
YEAR = 2021
START_DATE = date(YEAR, 1, 1)
END_DATE = date(YEAR, 7, 4)

DAYS_PER_BLOCK = 5          # One quota for each continuous five-day window
SAMPLE_PER_STRATUM = 5      # Final number selected per keyword per block (give up this setting after data collection)
VALID_CANDIDATE_TARGET = 10  # Stop immediately after this many valid candidates
MAX_RAW_RESULTS_PER_STRATUM = 50  # Safety ceiling if valid candidates are scarce
DOWNLOAD_DELAY = 15          # Seconds between page requests

# Specify filtering criteria when searching.
KEYWORDS = [
    "婚姻", "结婚", "婚恋", "恋爱", "对象", "伴侣", "相亲", "单身",
    "不婚", "晚婚", "恐婚", "催婚", "婚姻登记", "彩礼", "离婚冷静期",
]

FICTION_TERMS = ["小说", "阅读全文", "完结", "大结局", "晋江文学", "txt", "全章节", "笔趣阁"]
EXCLUDE_TERMS = ["微博问答", "超话", "恋与深空", "恋与制作人", "微博正文", "穿越"]
CELEBRITY_TERMS = [
    "谢娜", "何炅", "杨幂", "Angelababy", "陈坤", "赵丽颖", "易烊千玺", "王源", "王俊凯",
    "姚晨", "张杰", "迪丽热巴", "唐嫣", "林心如", "邓超", "陈乔恩", "刘亦菲", "杨紫",
    "宋茜", "赵薇", "郭德纲", "林志颖", "胡歌", "范冰冰", "王力宏", "陈赫", "黄子韬",
    "鹿晗", "贾乃亮", "罗志祥", "黄晓明", "薛之谦", "杨洋", "李晨", "林俊杰", "韩庚",
    "林更新", "刘烨", "张艺兴", "高圆圆", "刘涛", "孙俪", "王珞丹", "佟丽娅", "关晓彤",
    "吴磊", "戚薇", "范玮琪", "郑恺", "王祖蓝", "马伊琍", "张靓颖", "蔡依林", "赵露思",
    "郭采洁", "周冬雨", "虞书欣", "章子怡", "白鹿", "李沁", "景甜", "倪妮", "张馨予",
    "白百合", "古力娜扎", "江疏影", "杨超越", "肖战", "王一博", "马嘉祺", "宋亚轩",
    "刘耀文", "丁程鑫", "严浩翔", "张真源", "贺峻霖",
]

if START_DATE > END_DATE:
    raise ValueError("START_DATE must not be later than END_DATE.")
if DAYS_PER_BLOCK < 1 or SAMPLE_PER_STRATUM < 1:
    raise ValueError("DAYS_PER_BLOCK and SAMPLE_PER_STRATUM must be positive.")
if VALID_CANDIDATE_TARGET < SAMPLE_PER_STRATUM:
    raise ValueError("VALID_CANDIDATE_TARGET must be at least SAMPLE_PER_STRATUM.")
if MAX_RAW_RESULTS_PER_STRATUM < VALID_CANDIDATE_TARGET:
    raise ValueError("MAX_RAW_RESULTS_PER_STRATUM must be at least VALID_CANDIDATE_TARGET.")

WEIBO_COOKIE = getpass("Paste your Weibo cookie here: ").strip()
if not WEIBO_COOKIE:
    raise ValueError("A Weibo cookie is required.")

PERIOD_TAG = f"{START_DATE.isoformat()}_to_{END_DATE.isoformat()}"
DATA_DIR = Path(f"/content/weibo_sample_{PERIOD_TAG}")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("Period:", START_DATE, "to", END_DATE, "| Keywords:", len(KEYWORDS))
print("Sampling interval:", DAYS_PER_BLOCK, "days | Final quota:", SAMPLE_PER_STRATUM)
print("Valid-candidate stopping target per stratum:", VALID_CANDIDATE_TARGET)
print("Raw-result safety ceiling per stratum:", MAX_RAW_RESULTS_PER_STRATUM)
print("Output directory:", DATA_DIR)


# Add an early-cleaning pipeline to the crawler.
# A parsed post is therefore written to the keyword CSV only if it passes the cleaning rules, including the celebrity-name exclusion list. 
import json

pipeline_source = f"""
import csv
import os
from pathlib import Path

import regex
from scrapy.exceptions import DropItem

FICTION_TERMS = {FICTION_TERMS!r}
EXCLUDE_TERMS = {EXCLUDE_TERMS!r}
CELEBRITY_TERMS = {CELEBRITY_TERMS!r}
FICTION_PATTERN = regex.compile("|".join(map(regex.escape, FICTION_TERMS)), regex.IGNORECASE)
EXCLUDE_PATTERN = regex.compile("|".join(map(regex.escape, EXCLUDE_TERMS)))
CELEBRITY_PATTERN = regex.compile("|".join(map(regex.escape, CELEBRITY_TERMS)), regex.IGNORECASE)
OTHER_SCRIPT = regex.compile(r"(?!\\p{{Script=Han}}|\\p{{Script=Latin}})\\p{{L}}")


class EarlyCleaningPipeline:
    def open_spider(self, spider):
        path = Path(os.environ["WEIBO_REMOVED_FILE"])
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        self.handle = path.open("a", encoding="utf-8-sig", newline="")
        self.writer = csv.writer(self.handle)
        if new_file:
            self.writer.writerow([
                "id", "keyword", "微博正文", "发布时间", "removal_reason"
            ])
        self.valid_target = int(os.environ.get("WEIBO_VALID_TARGET", "10"))
        self.accepted = 0
        self.seen_ids = set()
        self.prefix_earliest = {{}}

    def process_item(self, item, spider):
        if self.accepted >= self.valid_target:
            raise DropItem("valid_candidate_target_already_reached")
        post = item.get("weibo", {{}})
        body = (post.get("text") or "").strip()
        query_keyword = item.get("keyword", "")
        post_id = str(post.get("id", ""))
        prefix20 = body[:20]
        reasons = []
        if not body:
            reasons.append("blank_body")
        if query_keyword and query_keyword not in body:
            reasons.append("query_keyword_not_in_body:" + query_keyword)
        fiction = FICTION_PATTERN.search(body)
        excluded = EXCLUDE_PATTERN.search(body)
        celebrity = CELEBRITY_PATTERN.search(body)
        other = OTHER_SCRIPT.search(body)
        if fiction:
            reasons.append("fiction_term:" + fiction.group())
        if excluded:
            reasons.append("excluded_term:" + excluded.group())
        if celebrity:
            reasons.append("celebrity_name:" + celebrity.group())
        if other:
            reasons.append("other_script:" + other.group())
        if post_id and post_id in self.seen_ids:
            reasons.append("duplicate_id")
        if reasons:
            self.writer.writerow([
                post.get("id", ""), item.get("keyword", ""), body,
                post.get("created_at", ""), "; ".join(reasons)
            ])
            self.handle.flush()
            raise DropItem("early_cleaning:" + "; ".join(reasons))

        published_at = str(post.get("created_at", ""))
        if prefix20 in self.prefix_earliest:
            earliest = self.prefix_earliest[prefix20]
            # Keep an additional row only when it is earlier; it does not
            # consume another valid-candidate slot. Global processing later
            # retains this earliest version and removes the previous one.
            if published_at and (not earliest or published_at < earliest):
                self.prefix_earliest[prefix20] = published_at
                if post_id:
                    self.seen_ids.add(post_id)
                return item
            self.writer.writerow([
                post.get("id", ""), item.get("keyword", ""), body,
                published_at, "duplicate_prefix20_not_earliest"
            ])
            self.handle.flush()
            raise DropItem("early_cleaning:duplicate_prefix20_not_earliest")

        if post_id:
            self.seen_ids.add(post_id)
        self.prefix_earliest[prefix20] = published_at
        self.accepted += 1
        if self.accepted >= self.valid_target:
            spider.logger.info(
                "Reached valid-candidate target: %s", self.valid_target
            )
            spider.crawler.engine.close_spider(
                spider, reason="valid_candidate_target_reached"
            )
        return item

    def close_spider(self, spider):
        if hasattr(self, "handle"):
            self.handle.close()
"""
(FULL_REPO / "weibo" / "sampling_filter.py").write_text(
    pipeline_source, encoding="utf-8"
)

settings_path = FULL_REPO / "weibo" / "settings.py"
settings_text = settings_path.read_text(encoding="utf-8")
marker = "# --- COURSE SAMPLE PIPELINES ---"
settings_text = settings_text.split(marker)[0].rstrip() + f"""

{marker}
ITEM_PIPELINES = {{
    'weibo.sampling_filter.EarlyCleaningPipeline': 100,
    'weibo.pipelines.DuplicatesPipeline': 300,
    'weibo.pipelines.CsvPipeline': 301,
}}
"""
settings_path.write_text(settings_text, encoding="utf-8")
print("Early-cleaning pipeline installed.")


# Crawl until the valid-candidate target is reached in every five-day keyword stratum.
import csv
import itertools
import os
import subprocess
import time
from collections import deque
from datetime import date, timedelta

SOURCE_COLUMNS = [
    "id", "user_id", "微博正文", "发布时间"
]
SAMPLE_COLUMNS = ["sampling_year", "block_start", "block_end", "query_keyword"] + SOURCE_COLUMNS
LOG_COLUMNS = [
    "task_id", "block_start", "block_end", "keyword", "valid_candidate_target",
    "raw_result_safety_ceiling",
    "valid_candidates", "return_code", "elapsed_seconds", "status", "diagnostic_log",
]

CANDIDATE_FILE = DATA_DIR / "eligible_candidates.csv"
REMOVED_FILE = DATA_DIR / "removed_during_crawl.csv"
TASK_LOG = DATA_DIR / "collection_task_log.csv"
DEBUG_DIR = DATA_DIR / "crawler_logs"
DEBUG_DIR.mkdir(exist_ok=True)

def make_blocks(start_date, end_date, block_days):
    """Create continuous windows and merge a short final remainder backward."""
    blocks = []
    current = start_date
    while current <= end_date:
        block_end = min(current + timedelta(days=block_days - 1), end_date)
        blocks.append((current, block_end))
        current = block_end + timedelta(days=1)

    if len(blocks) > 1:
        last_start, last_end = blocks[-1]
        last_length = (last_end - last_start).days + 1
        if last_length < block_days:
            previous_start, _ = blocks[-2]
            blocks[-2] = (previous_start, last_end)
            blocks.pop()

    return [(start.isoformat(), end.isoformat()) for start, end in blocks]

def count_csv_rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)

def append_rows(path, fieldnames, rows):
    rows = list(rows)
    if not rows:
        return
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

completed = set()
if TASK_LOG.exists():
    with TASK_LOG.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "done":
                completed.add(row["task_id"])

blocks = make_blocks(START_DATE, END_DATE, DAYS_PER_BLOCK)
tasks = [(start, end, keyword) for start, end in blocks for keyword in KEYWORDS]
print("Five-day blocks:", len(blocks), "| Total keyword tasks:", len(tasks))
print("Target valid candidate slots:", len(tasks) * VALID_CANDIDATE_TARGET)

consecutive_zero_tasks = 0
for task_number, (block_start, block_end, keyword) in enumerate(tasks, start=1):
    task_id = f"{block_start}_{block_end}_{keyword}"
    if task_id in completed:
        print(f"[{task_number}/{len(tasks)}] Already completed: {task_id}")
        continue

    source_file = FULL_REPO / "结果文件" / keyword / f"{keyword}.csv"
    rows_before = count_csv_rows(source_file)
    diagnostic = DEBUG_DIR / f"{block_start}_{block_end}_{keyword}.log"

    environment = os.environ.copy()
    environment.update({
        "WEIBO_COOKIE": WEIBO_COOKIE,
        "WEIBO_KEYWORDS": json.dumps([keyword], ensure_ascii=False),
        "WEIBO_START_DATE": block_start,
        "WEIBO_END_DATE": block_end,
        "WEIBO_TYPE": "1",          # original posts only
        "WEIBO_CONTAIN_TYPE": "0",  # no media restriction
        "WEIBO_REGION": json.dumps(["全部"], ensure_ascii=False),
        "WEIBO_FURTHER_THRESHOLD": "46",
        # The early-cleaning pipeline normally stops first. This is only a safety ceiling.
        "WEIBO_LIMIT_RESULT": str(MAX_RAW_RESULTS_PER_STRATUM),
        "WEIBO_VALID_TARGET": str(VALID_CANDIDATE_TARGET),
        "WEIBO_FETCH_IP": "0",
        "WEIBO_REMOVED_FILE": str(REMOVED_FILE),
    })

    print(f"[{task_number}/{len(tasks)}] {block_start} to {block_end} | {keyword}", flush=True)
    started = time.monotonic()
    with diagnostic.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            ["scrapy", "crawl", "search",
             "-s", "LOG_LEVEL=INFO",
             "-s", f"DOWNLOAD_DELAY={DOWNLOAD_DELAY}",
             "-s", "RANDOMIZE_DOWNLOAD_DELAY=True",
             "-s", "CONCURRENT_REQUESTS=1"],
            cwd=FULL_REPO,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    new_valid_rows = []
    if source_file.exists():
        with source_file.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in itertools.islice(reader, rows_before, None):
                new_valid_rows.append({
                    "sampling_year": YEAR,
                    "block_start": block_start,
                    "block_end": block_end,
                    "query_keyword": keyword,
                    **{column: row.get(column, "") for column in SOURCE_COLUMNS},
                })
    append_rows(CANDIDATE_FILE, SAMPLE_COLUMNS, new_valid_rows)

    status = "done" if result.returncode == 0 else "failed"
    record = {
        "task_id": task_id,
        "block_start": block_start,
        "block_end": block_end,
        "keyword": keyword,
        "valid_candidate_target": VALID_CANDIDATE_TARGET,
        "raw_result_safety_ceiling": MAX_RAW_RESULTS_PER_STRATUM,
        "valid_candidates": len(new_valid_rows),
        "return_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "status": status,
        "diagnostic_log": str(diagnostic),
    }
    append_rows(TASK_LOG, LOG_COLUMNS, [record])
    print("  Eligible after early cleaning:", len(new_valid_rows),
          "| Return code:", result.returncode)

    if result.returncode != 0:
        with diagnostic.open(encoding="utf-8", errors="replace") as handle:
            print("".join(deque(handle, maxlen=40)))
        raise RuntimeError(f"Crawler failed. Review {diagnostic}")

    consecutive_zero_tasks = consecutive_zero_tasks + 1 if not new_valid_rows else 0
    if consecutive_zero_tasks >= 5:
        raise RuntimeError(
            "Five consecutive tasks produced zero eligible candidates. "
            "Check whether the cookie has expired and inspect the crawler logs before continuing."
        )

print("Candidate collection complete:", CANDIDATE_FILE)
print("Early-removal audit:", REMOVED_FILE)
print("Task log:", TASK_LOG)
