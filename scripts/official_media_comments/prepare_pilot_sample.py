#!/usr/bin/env python3
"""Screen weibo-search CSV files and create an auditable discussion sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import tomllib


NOISE_PATTERNS = {
    "advertisement": re.compile(
        r"优惠|折扣|下单|购买|客服|欢迎咨询|咨询购买|购买链接|旗舰店|代购|婚庆|婚纱摄影|"
        r"加微(?:信)?|私信我|公众号|公#众#号|扣666|后台.{0,5}回复|扫码|课程报名|"
        r"房源|招聘|征婚启事|交友群|脱单群|返现|领取|代理|招商|"
        r"催眠课|课程推广|推广奇门|学生.{0,8}(笔记|反馈)"
    ),
    "fiction_or_serial": re.compile(
        r"第[一二三四五六七八九十百千\d]+章|全文阅读|小说|番外|连载|大结局|完本|"
        r"笔趣阁|书名[:：]|主角[:：]|文段试读|短剧|替嫁|闪婚老公|仙门|夫君|侯府|"
        r"失忆文|ABO|alpha|omega|生米煮成.{0,3}熟饭"
    ),
    "fandom_or_entertainment": re.compile(
        r"超话|打榜|应援|控评|唯粉|嗑cp|磕cp|官配|角色人生|新剧|男主|女主|动漫|"
        r"原著|演员|演技|追剧|综艺|电影|电视剧|广播剧|同人|周边|漫画家|主役|"
        r"出道|伪背德|第\d+封.{0,4}来信|那个世界|角色设定"
    ),
    "event_or_history_not_current_discussion": re.compile(
        r"奉子成婚.{0,30}(19|20)\d{2}年|古代|皇帝|公主|王妃|太子|史载|墓志|考古"
    ),
    "automated_or_spam": re.compile(
        r"早安打卡|晚安打卡|每日一善|互粉|诚信互关|扩列|随机抽奖|转发抽|机器人"
    ),
    "astrology_or_metaphysics": re.compile(
        r"星盘|星座|正缘|八字|坤造|命格|命中.{0,8}(水|火|土|木|金)|面相|太岁|"
        r"好运播报|上上签|生肖属相|业力|前世|奇门遁甲|解梦|神兽|成精|算命"
    ),
    "commercial_service": re.compile(
        r"单身试管|试管.{0,8}(客户|咨询|机构)|客户一家|精哥|面试选了|当地红娘|"
        r"好医保|蚂蚁保|患儿案例|投保|保费|婚恋阻碍|小儿推拿|推拿课程"
    ),
    "matchmaking_profile": re.compile(
        r"\d{2}女.{0,20}未婚|有房有车.{0,30}希望对方|身高\d{3}|\d{3}cm以上|"
        r"父母体制内|市直公务员.{0,30}(未婚|有房)"
    ),
    "sample_specific_entertainment": re.compile(
        r"麦琳|李行亮|黄晓明|叶珂|覃海洋|花絮|片场|玫瑰少年|卡塔尔小王子|"
        r"再见爱人|磕糖|嗑糖|综艺节目"
    ),
}

DISCUSSION_PATTERNS = {
    "first_person": re.compile(r"我|我们|本人|身边人|朋友|同事|父母|家里人"),
    "attitude_or_intention": re.compile(
        r"想结婚|不想结婚|愿意结婚|不愿结婚|恐婚|不婚|催婚|婚姻观|"
        r"想生|不想生|不敢生|生不起|养不起|生育意愿|育儿压力|结婚意愿"
    ),
    "reasoning": re.compile(r"因为|所以|但是|然而|原因|问题在于|意味着|成本|压力|选择"),
    "question_or_deliberation": re.compile(r"为什么|怎么看|如何看待|大家觉得|你们觉得|是否|该不该|值不值得|[？?]"),
    "experience": re.compile(r"经历|相亲过|结婚后|生完|生了|带娃|养娃|育儿|婚后|恋爱中|单身|离婚后"),
}


def parse_date(value: str) -> date | None:
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def parse_count(value: str) -> int:
    text = (value or "").strip().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 0
    number = float(match.group())
    if "万" in text:
        number *= 10_000
    return int(number)


def pid(value: str, salt: str) -> str:
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()[:24]


def locate(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return row[name].strip()
    return ""


def normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+|O网页链接", "", text.casefold())
    return re.sub(r"[#@\s\W_]+", "", text)


def load_salt(config_path: Path) -> str:
    value = os.environ.get("WEIBO_DATA_SALT", "").strip()
    if value:
        return value
    path = config_path.resolve().parent / "private" / "pseudonym_salt.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def canonical_group(group: str) -> str:
    if group.startswith("marriage"):
        return "marriage"
    if group.startswith("fertility") or group.startswith("family"):
        return "fertility"
    return group


def topical_groups(hits: dict[str, list[str]]) -> set[str]:
    topics: set[str] = set()
    for category, words in hits.items():
        if category == "marriage":
            topics.add("marriage")
        elif category in {"fertility", "family_childrearing"}:
            topics.add("fertility")
        elif category == "negative_attitudinal":
            if "恐婚" in words:
                topics.add("marriage")
            if any(word != "恐婚" for word in words):
                topics.add("fertility")
        else:
            topics.add(canonical_group(category))
    return topics


def keyword_hits(text: str, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    folded = text.casefold()
    hits: dict[str, list[str]] = {}
    for group, words in groups.items():
        found = sorted({word for word in words if word and word.casefold() in folded})
        if found:
            hits[group] = found
    return hits


def score_post(text: str, hits: dict[str, list[str]], comments: int,
               reposts: int, likes: int) -> tuple[int, list[str]]:
    signals = [name for name, pattern in DISCUSSION_PATTERNS.items() if pattern.search(text)]
    score = min(3, sum(len(words) for words in hits.values()))
    length = len(normalize_text(text))
    score += 2 if 30 <= length <= 800 else (1 if length >= 15 else 0)
    score += min(3, len(signals))
    score += 3 if comments >= 20 else (2 if comments >= 5 else (1 if comments >= 1 else 0))
    score += 1 if reposts + likes >= 10 else 0
    return score, signals


def evaluate_post(text: str, hits: dict[str, list[str]], flags: list[str],
                  comments: int, reposts: int, likes: int,
                  min_text_chars: int, min_comments: int,
                  min_engagement: int, min_score: int) -> tuple[str, list[str], int, list[str]]:
    score, signals = score_post(text, hits, comments, reposts, likes)
    reasons = list(flags)
    if not hits:
        reasons.append("unrelated_no_keyword_in_text")
    if len(normalize_text(text)) < min_text_chars:
        reasons.append("text_too_short")
    if comments < min_comments and comments + reposts + likes < min_engagement:
        reasons.append("insufficient_interaction")
    if not signals:
        reasons.append("no_discussion_signal")
    if score < min_score:
        reasons.append("low_quality_score")
    decision = "eligible" if not reasons else "exclude"
    return decision, sorted(set(reasons)), score, signals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="weibo-search 的‘结果文件’目录或单个 CSV")
    parser.add_argument("--config", type=Path, default=Path("pilot_2024_2025.toml"))
    parser.add_argument("--output", type=Path, default=Path("pilot_data/high_quality_posts_for_review.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("pilot_data/screening_audit.csv"))
    parser.add_argument("--all-posts", action="store_true", help="输出所有合格帖子，不做月度分层限额")
    args = parser.parse_args()

    with args.config.open("rb") as fh:
        cfg = tomllib.load(fh)
    start = date.fromisoformat(cfg["pilot"]["start_date"])
    end = date.fromisoformat(cfg["pilot"]["end_date"])
    per_stratum = int(cfg["pilot"].get("posts_per_month_per_group", 20))
    screening = cfg.get("screening", {})
    min_text = int(screening.get("min_text_chars", 20))
    min_comments = int(screening.get("min_comments", 3))
    min_engagement = int(screening.get("min_total_engagement", 10))
    min_score = int(screening.get("min_quality_score", 6))
    manual_exclusions: dict[str, str] = {}
    manual_path_value = str(screening.get("manual_exclusions_file", "")).strip()
    if manual_path_value:
        manual_path = Path(manual_path_value)
        if not manual_path.is_absolute():
            manual_path = args.config.resolve().parent / manual_path
        if manual_path.exists():
            with manual_path.open(encoding="utf-8-sig", newline="") as fh:
                manual_exclusions = {
                    row["post_id"].strip(): row.get("reason", "manual_review").strip()
                    for row in csv.DictReader(fh) if row.get("post_id", "").strip()
                }
    salt = load_salt(args.config)
    if len(salt) < 16:
        raise SystemExit("请设置至少16字符的 WEIBO_DATA_SALT")

    keyword_groups = {k: list(v) for k, v in cfg["keywords"].items()}
    search_queries = cfg.get("search_queries", keyword_groups)
    query_group = {word: canonical_group(group) for group, words in search_queries.items() for word in words}
    allowed_queries = {word for words in search_queries.values() for word in words}
    official_ids = {str(a.get("uid", "")) for a in cfg.get("official_accounts", []) if a.get("uid")}

    files = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.csv"))
    unique: dict[str, dict[str, str]] = {}
    normalized_seen: dict[str, str] = {}
    for path in files:
        query = path.stem
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                post_id = locate(row, "id", "微博id")
                created_raw = locate(row, "发布时间", "created_at")
                created = parse_date(created_raw)
                if not post_id or not created or not (start <= created <= end):
                    continue
                text = locate(row, "微博正文", "text")
                query = locate(row, "keyword", "关键词") or path.stem
                if query not in allowed_queries:
                    continue
                raw_uid = locate(row, "user_id", "用户id")
                auth = locate(row, "user_authentication", "用户认证")
                hits = keyword_hits(text, keyword_groups)
                canonical_hits = sorted(topical_groups(hits))
                group = "both" if len(canonical_hits) > 1 else (canonical_hits[0] if canonical_hits else query_group.get(query, "unmapped"))
                comments = parse_count(locate(row, "评论数", "comments_count"))
                reposts = parse_count(locate(row, "转发数", "reposts_count"))
                likes = parse_count(locate(row, "点赞数", "attitudes_count"))
                flags = [name for name, pattern in NOISE_PATTERNS.items() if pattern.search(text)]
                if post_id in manual_exclusions:
                    flags.append("manual_exclusion_" + manual_exclusions[post_id])
                query_terms = query.split()
                if not all(term in text for term in query_terms):
                    flags.append("search_query_not_in_text")
                if text.lstrip().startswith("【") or text.lstrip().startswith("#微博附注#"):
                    flags.append("news_repost_style")
                fertility_context = re.search(
                    r"生育|孩子|小孩|娃|婴儿|怀孕|孕育|二孩|三孩|二胎|三胎|"
                    r"出生率|生育率|带娃|育儿|母婴|婚育", text
                )
                if query in {"养不起", "生不起"} and not fertility_context:
                    flags.append("ambiguous_term_without_fertility_context")
                hashtags = re.findall(r"#([^#]+)#", text)
                body_without_hashtags = re.sub(r"#[^#]+#", " ", text)
                body_hits = keyword_hits(body_without_hashtags, keyword_groups)
                meaningful_body_hits = {
                    word for words in body_hits.values() for word in words
                    if word not in {"家庭", "对象", "伴侣", "恋爱", "单身", "养不起", "生不起"}
                }
                if hashtags and not meaningful_body_hits:
                    flags.append("topic_only_in_hashtag")
                if auth.lower() in {"蓝v", "blue_v"} or raw_uid in official_ids:
                    flags.append("official_account_in_user_corpus")
                normalized = normalize_text(text)
                if normalized and normalized in normalized_seen and normalized_seen[normalized] != post_id:
                    flags.append("duplicate_text")
                elif normalized:
                    normalized_seen[normalized] = post_id
                decision, reasons, score, signals = evaluate_post(
                    text, hits, flags, comments, reposts, likes,
                    min_text, min_comments, min_engagement, min_score,
                )
                item = {
                    "source_group": "user_initiated", "post_id": post_id,
                    "bid": locate(row, "bid", "微博bid"), "created_at": created_raw,
                    "month": created.strftime("%Y-%m"), "keyword_group": group,
                    "keyword_categories": "|".join(sorted(hits)),
                    "matched_keywords": "|".join(sorted({w for words in hits.values() for w in words})),
                    "search_queries": query, "text": text, "author_pid": pid(raw_uid, salt),
                    "author_authentication": auth, "comments_count": str(comments),
                    "reposts_count": str(reposts), "attitudes_count": str(likes),
                    "discussion_signals": "|".join(signals), "quality_score": str(score),
                    "quality_tier": "high" if score >= 9 else ("medium" if score >= min_score else "low"),
                    "screening_decision": decision, "exclusion_reasons": "|".join(reasons),
                    "screening_flags": "|".join(sorted(set(flags))),
                    "review_status": "pending" if decision == "eligible" else "exclude", "review_note": "",
                }
                if post_id in unique:
                    old = unique[post_id]
                    old["search_queries"] = "|".join(sorted(set(old["search_queries"].split("|")) | {query}))
                    old["matched_keywords"] = "|".join(sorted(
                        {w for w in old["matched_keywords"].split("|") if w} |
                        {w for w in item["matched_keywords"].split("|") if w}
                    ))
                else:
                    unique[post_id] = item

    audit_rows = sorted(unique.values(), key=lambda x: (x["month"], x["post_id"]))
    eligible = [row for row in audit_rows if row["screening_decision"] == "eligible"]
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for item in eligible:
        strata[(item["month"], item["keyword_group"])].append(item)
    selected: list[dict[str, str]] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda x: (
            -int(x["quality_score"]),
            hashlib.sha256(("quality-v1|" + x["post_id"]).encode()).hexdigest(),
        ))
        selected.extend(rows if args.all_posts else rows[:per_stratum])
    selected.sort(key=lambda x: (x["month"], x["keyword_group"], -int(x["quality_score"]), x["post_id"]))

    fields = list(audit_rows[0].keys()) if audit_rows else ["post_id"]
    for path, rows in ((args.audit_output, audit_rows), (args.output, selected)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    counts = Counter(reason for row in audit_rows for reason in row["exclusion_reasons"].split("|") if reason)
    print(f"读取 {len(files)} 个文件，去重后 {len(audit_rows)} 条，自动合格 {len(eligible)} 条，输出复核 {len(selected)} 条")
    print(f"筛选审计：{args.audit_output}")
    print(f"高质量候选：{args.output}")
    if counts:
        print("主要排除原因：" + "；".join(f"{k}={v}" for k, v in counts.most_common(8)))
    print("请人工复核候选，将纳入行的 review_status 从 pending 改为 include 后再采评论。")


if __name__ == "__main__":
    main()
