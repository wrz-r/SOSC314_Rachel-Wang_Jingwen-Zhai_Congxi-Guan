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
