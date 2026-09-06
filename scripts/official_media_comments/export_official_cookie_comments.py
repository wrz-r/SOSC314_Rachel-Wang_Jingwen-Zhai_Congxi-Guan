#!/usr/bin/env python3
"""Create a minimal, de-identified official-media comment CSV from the cookie pilot."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from datetime import datetime
from pathlib import Path


AD_OR_SPAM = re.compile(r"加微(?:信)?|公众号|优惠|下单|客服|返现|推广|互粉|回关|打卡|扩列|抽奖|扫码|二维码|点击链接")
NON_TEXT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
EMOTICON = re.compile(r"\[[^\]\r\n]{1,40}\]")

FIELDS = [
    "official_account", "parent_post_id", "post_created_at", "post_month", "keyword_group",
    "keyword_categories", "matched_keywords", "parent_post_text", "comment_id",
    "comment_created_at", "comment_month", "comment_text",
]


def iso_time(value: object) -> str:
    text = str(value or "").strip()
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            pass
    return text


def normalized(text: object) -> str:
    # 微博把表情编码成 [good]、[打call] 等；它们不是可分析的文字评论。
    without_emoticons = EMOTICON.sub("", str(text or ""))
    return NON_TEXT.sub("", without_emoticons).casefold()


def reason(text: object) -> str:
    clean = normalized(text)
    if len(clean) < 8:
        return "too_short_or_nontext"
    if AD_OR_SPAM.search(str(text or "")):
        return "advertising_or_spam"
    if len(clean) >= 8 and len(set(clean)) <= 2:
        return "repetitive_text"
    return ""


def write(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("posts", type=Path)
    parser.add_argument("comments_db", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pilot_data/official_media_2025_cookie/official_media_comments_2025.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("pilot_data/official_media_2025_cookie/official_comment_screening_audit.csv"))
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    args = parser.parse_args()
    allowed = {"include", "pending"} if args.include_pending else {"include"}
    with args.posts.open(encoding="utf-8-sig", newline="") as fh:
        posts = {row["post_id"]: row for row in csv.DictReader(fh) if row.get("review_status", "").lower() in allowed}
    if not posts:
        raise SystemExit("没有通过人工审核的母帖；先将 review_status 改为 include，或明确使用 --include-pending。")
    if not args.comments_db.exists():
        raise SystemExit(f"评论数据库不存在：{args.comments_db}")
    start_date, end_date = args.start_date, args.end_date
    db = sqlite3.connect(args.comments_db)
    db.row_factory = sqlite3.Row
    seen: set[str] = set()
    retained: dict[str, list[sqlite3.Row]] = {post_id: [] for post_id in posts}
    audit: list[dict[str, str]] = []
    for comment in db.execute("SELECT comment_id,post_id,created_at,text FROM comments ORDER BY post_id,retrieval_rank,comment_id"):
        post = posts.get(str(comment["post_id"]))
        if not post:
            continue
        timestamp = iso_time(comment["created_at"])
        exclusion = reason(comment["text"])
        if not timestamp or not (start_date <= timestamp[:10] <= end_date):
            exclusion = "outside_comment_time_window"
        compact = normalized(comment["text"])
        raw_text = str(comment["text"] or "").strip()
        if not exclusion and re.fullmatch(r"#[^#\r\n]+#", raw_text):
            exclusion = "hashtag_only"
        parent_compact = normalized(post.get("text", ""))
        if not exclusion and len(compact) >= 12 and compact in parent_compact:
            exclusion = "repeats_parent_post_without_commentary"
        if not exclusion and compact in seen:
            exclusion = "duplicate_comment_text"
        audit.append({
            "comment_id": str(comment["comment_id"]), "parent_post_id": str(comment["post_id"]),
            "comment_created_at": timestamp, "decision": "exclude" if exclusion else "include",
            "reason": exclusion, "comment_text": str(comment["text"]),
        })
        if exclusion:
            continue
        seen.add(compact)
        retained[str(comment["post_id"])].append(comment)
    final: list[dict[str, str]] = []
    for post_id, post in posts.items():
        comments = retained.get(post_id, [])
        if not comments:
            continue
        for index, comment in enumerate(comments):
            timestamp = iso_time(comment["created_at"])
            # CSV 无法合并单元格：每组第一行保留母帖，后续行只保留关系 ID 和评论。
            first = index == 0
            final.append({
                "official_account": post.get("official_account", "") if first else "",
                "parent_post_id": post["post_id"],
                "post_created_at": post.get("created_at", "") if first else "",
                "post_month": post.get("month", "") if first else "",
                "keyword_group": post.get("keyword_group", "") if first else "",
                "keyword_categories": post.get("keyword_categories", "") if first else "",
                "matched_keywords": post.get("matched_keywords", "") if first else "",
                "parent_post_text": post.get("text", "") if first else "",
                "comment_id": str(comment["comment_id"]),
                "comment_created_at": timestamp,
                "comment_month": timestamp[:7],
                "comment_text": str(comment["text"]),
            })
    db.close()
    write(args.output, FIELDS, final)
    write(args.audit_output, ["comment_id", "parent_post_id", "comment_created_at", "decision", "reason", "comment_text"], audit)
    print(f"输出 {len(final)} 条高质量评论：{args.output}")


if __name__ == "__main__":
    main()
