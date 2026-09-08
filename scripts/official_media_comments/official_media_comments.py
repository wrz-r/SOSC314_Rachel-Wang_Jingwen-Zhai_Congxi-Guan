#!/usr/bin/env python3
"""Simple pipeline for collecting comments under official-media Weibo posts.

Workflow:
1. Search Weibo posts by keywords with dataabc/weibo-search.
2. Keep relevant posts from selected official-media accounts.
3. Collect public top-level comments under each post.
4. Save one comment per row in a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


# Research keywords used to search relevant Weibo posts.
KEYWORDS = {
    "marriage": [
        "结婚", "婚姻", "婚恋", "恋爱", "相亲", "单身", "不婚",
        "晚婚", "恐婚", "彩礼", "离婚", "催婚", "婚姻登记",
    ],
    "fertility": [
        "生育", "生孩子", "生娃", "出生率", "生育率", "二孩", "三孩",
        "二胎", "三胎", "育儿", "养娃", "生育意愿", "不想生",
        "不敢生", "生不起", "养不起",
    ],
}

# Exact UID whitelist prevents non-official accounts from entering the sample.
OFFICIAL_ACCOUNTS = {
    "1663072851": "China Daily",
    "2656274875": "CCTV News",
    "2803301701": "People's Daily",
    "1699432410": "Xinhua News Agency",
    "1784473157": "China News Service",
    "2834480301": "Healthy China",
}

OUTPUT_FIELDS = [
    "official_account", "parent_post_id", "post_created_at",
    "keyword_group", "matched_keywords", "parent_post_text",
    "comment_id", "comment_created_at", "comment_text",
]

SPAM = re.compile(r"加微(?:信)?|公众号|优惠|下单|客服|返现|推广|互粉|扫码|二维码|点击链接")
HTML_TAG = re.compile(r"<[^>]+>")
NON_TEXT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def get_value(row: dict[str, str], *names: str) -> str:
    """Read a field from either Chinese or English weibo-search columns."""
    for name in names:
        if row.get(name):
            return str(row[name]).strip()
    return ""


def clean_text(value: object) -> str:
    """Remove HTML tags from post and comment text."""
    text = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", HTML_TAG.sub("", text)).strip()


def parse_time(value: object) -> str:
    """Convert Weibo time to a consistent ISO-style time where possible."""
    text = str(value or "").strip()
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return text


def search_posts(repo: Path, cookie: str, start: str, end: str) -> None:
    """Search media posts by keywords using dataabc/weibo-search."""
    if not (repo / "scrapy.cfg").exists():
        raise SystemExit(f"Not a weibo-search directory: {repo}")

    all_keywords = [word for group in KEYWORDS.values() for word in group]
    environment = os.environ.copy()
    environment.update({
        "WEIBO_COOKIE": cookie,
        "WEIBO_KEYWORDS": json.dumps(all_keywords, ensure_ascii=False),
        "WEIBO_START_DATE": start,
        "WEIBO_END_DATE": end,
        "WEIBO_TYPE": "5",       # 5 = media posts in weibo-search
        "WEIBO_FETCH_IP": "0",
    })

    command = [
        sys.executable, "-m", "scrapy", "crawl", "search",
        "-s", "DOWNLOAD_DELAY=15",
        "-s", "CONCURRENT_REQUESTS=1",
    ]
    subprocess.run(command, cwd=repo, env=environment, check=True)


def select_official_posts(
    results_dir: Path, start: str, end: str, posts_csv: Path
) -> list[dict[str, str]]:
    """Keep relevant posts from the predefined official-account whitelist."""
    selected: dict[str, dict[str, str]] = {}

    for path in results_dir.rglob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                post_id = get_value(row, "微博id", "id", "post_id")
                uid = get_value(row, "用户id", "user_id", "uid")
                created_at = get_value(row, "发布时间", "created_at")
                text = clean_text(get_value(row, "微博正文", "text"))

                if not post_id or uid not in OFFICIAL_ACCOUNTS:
                    continue
                if not created_at or not (start <= created_at[:10] <= end):
                    continue

                hits = {
                    group: [word for word in words if word in text]
                    for group, words in KEYWORDS.items()
                }
                hits = {group: words for group, words in hits.items() if words}
                if not hits or len(text) < 15:
                    continue

                selected[post_id] = {
                    "official_account": OFFICIAL_ACCOUNTS[uid],
                    "post_id": post_id,
                    "created_at": created_at,
                    "keyword_group": "both" if len(hits) > 1 else next(iter(hits)),
                    "matched_keywords": "|".join(
                        sorted({word for words in hits.values() for word in words})
                    ),
                    "text": text,
                }

    posts = sorted(selected.values(), key=lambda row: row["created_at"])
    write_csv(posts_csv, list(posts[0]) if posts else ["post_id"], posts)
    print(f"Selected {len(posts)} official-media posts.")
    return posts


def good_comment(text: str, seen: set[str]) -> bool:
    """Remove very short, duplicate, and obvious advertising comments."""
    compact = NON_TEXT.sub("", text).casefold()
    if len(compact) < 8 or compact in seen or SPAM.search(text):
        return False
    seen.add(compact)
    return True


def collect_comments(
    posts: list[dict[str, str]], cookie: str, limit: int, delay: float
) -> list[dict[str, str]]:
    """Collect public top-level comments from Weibo's web comment endpoint."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": cookie,
        "Referer": "https://weibo.com/",
        "Accept": "application/json, text/plain, */*",
    }
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    for post in posts:
        max_id = "0"
        saved = 0
        first_comment = True

        while saved < limit:
            params = {
                "is_reload": "1", "id": post["post_id"],
                "is_show_bulletin": "2", "is_mix": "0",
                "count": "20", "max_id": max_id,
            }
            url = "https://weibo.com/ajax/statuses/buildComments?" + urllib.parse.urlencode(params)
            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))

            comments = payload.get("data") or []
            for comment in comments:
                if saved >= limit:
                    break
                comment_text = clean_text(comment.get("text_raw") or comment.get("text"))
                if not good_comment(comment_text, seen):
                    continue

                # Show parent-post information only once for each comment block.
                output.append({
                    "official_account": post["official_account"] if first_comment else "",
                    "parent_post_id": post["post_id"],
                    "post_created_at": post["created_at"] if first_comment else "",
                    "keyword_group": post["keyword_group"] if first_comment else "",
                    "matched_keywords": post["matched_keywords"] if first_comment else "",
                    "parent_post_text": post["text"] if first_comment else "",
                    "comment_id": str(comment.get("id") or ""),
                    "comment_created_at": parse_time(comment.get("created_at")),
                    "comment_text": comment_text,
                })
                saved += 1
                first_comment = False

            next_id = str(payload.get("max_id") or "0")
            if not comments or next_id == "0":
                break
            max_id = next_id
            time.sleep(delay)

        print(f"Post {post['post_id']}: {saved} comments.")

    return output


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Save UTF-8 CSV output for later sentiment analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weibo-search", type=Path, required=True)
    parser.add_argument("--cookie-file", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-comments", type=int, default=100)
    parser.add_argument("--delay", type=float, default=8.0)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--posts-output", type=Path, default=Path("official_posts.csv"))
    parser.add_argument("--output", type=Path, default=Path("official_media_comments.csv"))
    args = parser.parse_args()

    cookie = args.cookie_file.read_text(encoding="utf-8").strip()
    if not cookie:
        raise SystemExit("The local Cookie file is empty.")

    # Step 1: retrieve posts by keyword and date.
    if not args.skip_search:
        search_posts(args.weibo_search, cookie, args.start, args.end)

    # Step 2: keep relevant posts from selected official accounts.
    posts = select_official_posts(
        args.weibo_search / "结果文件", args.start, args.end, args.posts_output
    )

    # Step 3: collect public comments under each selected post.
    comments = collect_comments(posts, cookie, args.max_comments, args.delay)

    # Step 4: export one comment per row for sentiment analysis.
    write_csv(args.output, OUTPUT_FIELDS, comments)
    print(f"Saved {len(comments)} comments to {args.output}")


if __name__ == "__main__":
    main()
