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






