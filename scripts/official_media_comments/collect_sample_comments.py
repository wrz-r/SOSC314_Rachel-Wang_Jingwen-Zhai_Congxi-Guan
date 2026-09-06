#!/usr/bin/env python3
"""Collect top-level comments for manually approved pilot posts.

Uses Weibo's web AJAX endpoint conservatively. Run only with authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
import sqlite3
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import tomllib

try:
    import certifi
except ModuleNotFoundError:  # pragma: no cover - available in project venv
    certifi = None


def pseudonym(value: object, salt: str) -> str:
    return hmac.new(salt.encode(), str(value).encode(), hashlib.sha256).hexdigest()[:24]


def load_local_cookie(base: Path) -> str:
    value = os.environ.get("WEIBO_COOKIE", "")
    if value:
        return value.strip()
    path = base / "private" / "weibo_cookie.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def load_salt(base: Path) -> str:
    value = os.environ.get("WEIBO_DATA_SALT", "").strip()
    if value:
        return value
    path = base / "private" / "pseudonym_salt.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def parse_display_count(value: str) -> int:
    text = (value or "").strip().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 0
    number = float(match.group())
    if "万" in text:
        number *= 10_000
    return int(number)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--config", type=Path, default=Path("pilot_2024_2025.toml"))
    parser.add_argument("--database", type=Path, default=Path("pilot_data/pilot_comments.sqlite3"))
    parser.add_argument("--include-pending", action="store_true",
                        help="同时处理尚未人工审核的 pending 帖子")
    parser.add_argument("--max-posts", type=int, default=0,
                        help="本次最多处理多少个母帖；0表示不限")
    parser.add_argument("--max-comments", type=int,
                        help="覆盖配置中的每帖评论上限；0表示抓取全部可访问一级评论")
    parser.add_argument("--min-reported-comments", type=int, default=0,
                        help="只处理搜索结果中报告评论数不低于该值的母帖")
    args = parser.parse_args()
    if os.environ.get("WEIBO_RESEARCH_AUTHORIZED") != "YES":
        raise SystemExit("采集前请完成授权/伦理审查，并设置 WEIBO_RESEARCH_AUTHORIZED=YES")
    cookie = load_local_cookie(args.config.resolve().parent)
    salt = load_salt(args.config.resolve().parent)
    if not cookie:
        raise SystemExit("缺少 WEIBO_COOKIE 或 private/weibo_cookie.txt；不要把 Cookie 发送到聊天")
    if len(salt) < 16:
        raise SystemExit("请设置至少16字符的 WEIBO_DATA_SALT")
    with args.config.open("rb") as fh:
        cfg = tomllib.load(fh)["pilot"]
    limit = args.max_comments if args.max_comments is not None else int(cfg.get("max_comments_per_post", 100))
    if limit < 0:
        raise SystemExit("--max-comments 不能小于0")
    interval = float(cfg.get("comment_request_interval_seconds", 8))

    allowed = {"include", "pending"} if args.include_pending else {"include"}
    with args.sample.open(encoding="utf-8-sig", newline="") as fh:
        posts = [
            r for r in csv.DictReader(fh)
            if r.get("review_status", "").strip().lower() in allowed
            and parse_display_count(r.get("comments_count", "")) >= args.min_reported_comments
        ]
    if not posts:
        raise SystemExit("样本中没有 review_status=include 的帖子")

    args.database.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.database)
    db.executescript("""
      CREATE TABLE IF NOT EXISTS comments(
        comment_id TEXT PRIMARY KEY, post_id TEXT NOT NULL, created_at TEXT,
        text TEXT NOT NULL, author_pid TEXT, like_count INTEGER,
        retrieval_rank INTEGER, collected_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS progress(
        post_id TEXT PRIMARY KEY, next_max_id TEXT, saved_count INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL, updated_at TEXT NOT NULL);
    """)
    headers = {
        "User-Agent": "Mozilla/5.0 (Academic research pilot; contact project owner)",
        "Cookie": cookie,
        "Referer": "https://weibo.com/",
        "Accept": "application/json, text/plain, */*",
    }
    ssl_context = ssl.create_default_context(cafile=certifi.where() if certifi else None)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ssl_context)
    )
    processed_posts = 0
    try:
        for post in posts:
            post_id = post["post_id"]
            state = db.execute("SELECT next_max_id,saved_count,status FROM progress WHERE post_id=?", (post_id,)).fetchone()
            if state:
                if state[2] == "exhausted":
                    continue
                if state[2] == "capped" and limit > 0 and state[1] >= limit:
                    continue
            if args.max_posts and processed_posts >= args.max_posts:
                break
            processed_posts += 1
            max_id, saved = (state[0], state[1]) if state else ("0", 0)
            while limit == 0 or saved < limit:
                params = {
                    "is_reload": "1", "id": post_id, "is_show_bulletin": "2",
                    "is_mix": "0", "count": "20", "max_id": max_id or "0",
                }
                url = "https://weibo.com/ajax/statuses/buildComments?" + urllib.parse.urlencode(params)
                req = urllib.request.Request(url, headers=headers)
                try:
                    with opener.open(req, timeout=30) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    if exc.code in (403, 418, 429):
                        raise SystemExit(f"帖子 {post_id} 返回 HTTP {exc.code}，已停止；请检查授权或频率限制")
                    raise
                except urllib.error.URLError as exc:
                    raise SystemExit(f"帖子 {post_id} 网络连接失败，已安全停止且保留进度：{exc.reason}") from exc
                except TimeoutError as exc:
                    raise SystemExit(f"帖子 {post_id} 读取超时，已安全停止且保留进度；稍后重运行会继续未完成页。") from exc
                rows = payload.get("data") or []
                for obj in rows:
                    if limit and saved >= limit:
                        break
                    user = obj.get("user") or {}
                    text = obj.get("text_raw") or obj.get("text") or ""
                    cid = str(obj.get("id") or "")
                    if not cid:
                        continue
                    saved += 1
                    db.execute(
                        "INSERT OR IGNORE INTO comments VALUES (?,?,?,?,?,?,?,?)",
                        (cid, post_id, obj.get("created_at"), text,
                         pseudonym(user.get("idstr", user.get("id", "")), salt),
                         obj.get("like_counts", obj.get("like_count")), saved,
                         datetime.now(timezone.utc).isoformat()),
                    )
                next_id = str(payload.get("max_id") or "0")
                exhausted = not rows or next_id == "0"
                capped = limit > 0 and saved >= limit
                status = "exhausted" if exhausted else ("capped" if capped else "running")
                db.execute(
                    "INSERT OR REPLACE INTO progress VALUES (?,?,?,?,?)",
                    (post_id, next_id, saved, status,
                     datetime.now(timezone.utc).isoformat()),
                )
                db.commit()
                print(f"{post_id}: {saved} 条")
                if exhausted or capped:
                    break
                max_id = next_id
                time.sleep(interval)
    finally:
        db.close()


if __name__ == "__main__":
    main()
