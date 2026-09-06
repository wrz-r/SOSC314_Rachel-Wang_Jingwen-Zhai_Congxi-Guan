#!/usr/bin/env python3
"""Filter authorised weibo-search results to predeclared official accounts."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import date
from pathlib import Path

import tomllib

from prepare_pilot_sample import (
    NOISE_PATTERNS, canonical_group, keyword_hits, locate, normalize_text,
    parse_count, parse_date, topical_groups,
)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="weibo-search 的结果文件目录")
    parser.add_argument("--config", type=Path, default=Path("official_media_2025_cookie.toml"))
    parser.add_argument("--output", type=Path, default=Path("pilot_data/official_media_2025_cookie/official_posts_for_comments.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("pilot_data/official_media_2025_cookie/official_post_screening_audit.csv"))
    args = parser.parse_args()
    with args.config.open("rb") as fh:
        cfg = tomllib.load(fh)
    start = date.fromisoformat(cfg["pilot"]["start_date"])
    end = date.fromisoformat(cfg["pilot"]["end_date"])
    screening = cfg.get("screening", {})
    min_text = int(screening.get("min_parent_text_chars", 15))
    min_comments = int(screening.get("min_reported_comments", 3))
    min_engagement = int(screening.get("min_total_engagement", 10))
    issue_markers = [str(marker).casefold() for marker in screening.get("official_issue_markers", [])]
    groups = {key: list(value) for key, value in cfg["keywords"].items()}
    searches = cfg["search_queries"]
    allowed_queries = {word for words in searches.values() for word in words}
    query_group = {word: canonical_group(group) for group, words in searches.items() for word in words}
    accounts = {str(row["uid"]): str(row["label"]) for row in cfg["official_accounts"]}
    manual_path = Path(str(screening.get("manual_exclusions_file", "")))
    if not manual_path.is_absolute():
        manual_path = args.config.parent / manual_path
    manual = set()
    if manual_path.exists():
        with manual_path.open(encoding="utf-8-sig", newline="") as fh:
            manual = {row["post_id"].strip() for row in csv.DictReader(fh) if row.get("post_id", "").strip()}

    kept: dict[str, dict[str, str]] = {}
    for path in sorted(args.input.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                post_id = locate(raw, "id", "微博id")
                created_at = locate(raw, "发布时间", "created_at")
                created = parse_date(created_at)
                query = locate(raw, "keyword", "关键词") or path.stem
                uid = locate(raw, "user_id", "用户id")
                if not post_id or not created or not (start <= created <= end) or query not in allowed_queries or uid not in accounts:
                    continue
                text = locate(raw, "微博正文", "text")
                comments = parse_count(locate(raw, "评论数", "comments_count"))
                reposts = parse_count(locate(raw, "转发数", "reposts_count"))
                likes = parse_count(locate(raw, "点赞数", "attitudes_count"))
                hits = keyword_hits(text, groups)
                categories = sorted(hits)
                broad = sorted(topical_groups(hits))
                keyword_group = "both" if len(broad) > 1 else (broad[0] if broad else query_group[query])
                flags = [name for name in ("advertisement", "fiction_or_serial", "fandom_or_entertainment", "automated_or_spam") if NOISE_PATTERNS[name].search(text)]
                if post_id in manual:
                    flags.append("manual_exclusion")
                if not hits:
                    flags.append("no_topic_keyword")
                if len(normalize_text(text)) < min_text:
                    flags.append("parent_text_too_short")
                if locate(raw, "retweet_id"):
                    flags.append("official_repost")
                if comments < min_comments and comments + reposts + likes < min_engagement:
                    flags.append("insufficient_parent_discussion")
                if issue_markers and not any(marker in text.casefold() for marker in issue_markers):
                    flags.append("not_clear_marriage_fertility_social_issue")
                record = {
                    "source_group": "official", "official_account": accounts[uid], "post_id": post_id,
                    "created_at": created_at, "month": created.strftime("%Y-%m"), "keyword_group": keyword_group,
                    "keyword_categories": "|".join(categories),
                    "matched_keywords": "|".join(sorted({word for words in hits.values() for word in words})),
                    "search_queries": query, "text": text, "comments_count": str(comments),
                    "reposts_count": str(reposts), "attitudes_count": str(likes),
                    "screening_decision": "eligible" if not flags else "exclude",
                    "exclusion_reasons": "|".join(sorted(set(flags))),
                    "review_status": "pending" if not flags else "exclude", "review_note": "",
                }
                if post_id in kept:
                    old = kept[post_id]
                    old["search_queries"] = "|".join(sorted(set(old["search_queries"].split("|")) | {query}))
                    old["matched_keywords"] = "|".join(sorted(set(old["matched_keywords"].split("|")) | set(record["matched_keywords"].split("|"))))
                else:
                    kept[post_id] = record

    rows = sorted(kept.values(), key=lambda row: (row["month"], row["official_account"], row["post_id"]))
    eligible = [row for row in rows if row["screening_decision"] == "eligible"]
    headers = list(rows[0]) if rows else ["post_id"]
    write_csv(args.audit_output, headers, rows)
    write_csv(args.output, headers, eligible)
    exclusions = Counter(reason for row in rows for reason in row["exclusion_reasons"].split("|") if reason)
    print(f"官方账号候选 {len(rows)} 条，自动合格 {len(eligible)} 条。")
    if exclusions:
        print("主要排除原因：" + "；".join(f"{key}={value}" for key, value in exclusions.most_common(6)))
    print("请抽检合格母帖；确认后可将 review_status 改为 include，或在授权范围内用 --include-pending 试采。")


if __name__ == "__main__":
    main()
