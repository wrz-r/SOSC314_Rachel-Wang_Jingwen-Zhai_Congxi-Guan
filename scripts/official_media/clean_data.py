
import argparse
import csv
import hashlib
import html
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


# 1. Settings:
YEAR_TO_CLEAN = 2023                 
SOURCES = {
    "people.com.cn": "People.cn",
    "cctv.com": "CCTV",
    "chinanews.com.cn": "China News Service",
    "chinanews.com": "China News Service",
}
# Clear marriage-related words.
STRONG_KEYWORDS = [
    "婚姻", "婚姻登记", "结婚登记", "结婚", "彩礼", "高价彩礼", "婚俗",
    "婚恋", "晚婚", "不婚", "恐婚", "催婚", "离婚",
    "离婚冷静期", "结婚率", "初婚",
]
# These words need nearby marriage context.
WEAK_KEYWORDS = ["对象", "单身", "恋爱", "相亲", "伴侣"]

CONTEXT_KEYWORDS = [
    "婚姻", "婚恋", "结婚", "恋爱", "男女", "青年", "择偶", "伴侣",
    "情侣", "夫妻", "配偶", "家庭", "彩礼", "交友", "婚介",
]
# Ignore these unrelated expressions during screening.
FALSE_POSITIVE_PATTERNS = [
    "表彰对象", "救助对象", "帮扶对象", "优抚对象", "服务对象", "保障对象",
    "调查对象", "检查对象", "巡视对象", "筛查对象", "监管对象", "管理对象",
    "研究对象", "采访对象", "适用对象", "招生对象", "培训对象", "资助对象",
    "补贴对象", "征收对象", "评选对象",
    "面向对象", "对象存储", "对象识别", "操作对象",
    "山水相亲", "人文相亲", "相知相亲", "血缘相亲", "民心相亲",
    "动物恋爱", "恋爱季节", "宠物伴侣", "伴侣动物", "咖啡伴侣", "音乐伴侣",
]

# Possible entertainment or advertising: check manually.
ENTERTAINMENT_TERMS = [
    "电视剧", "电影", "综艺", "演员", "剧情", "短剧", "微短剧", "影视", "小说",
    "剧中", "角色",
]
COMMERCIAL_TERMS = [
    "婚庆促销", "婚纱摄影套餐", "婚宴套餐", "限时优惠", "优惠券", "团购",
    "招商加盟", "品牌推广", "广告推广",
]

# Short text needs review; body relevance needs 3 sentences and 20% coverage.
MIN_TEXT_LENGTH = 200                 # Non-whitespace characters; shorter = review.
MIN_RELEVANT_SENTENCES = 3             # At least 3 relevant body sentences/chunks.
MIN_RELEVANT_RATIO = 0.20              # At least 20% of body chunks must be relevant.

# Cut known People.cn footer blocks, not ordinary mentions of the website.
PEOPLE_FOOTER_RE = re.compile(
    r"人民日报社概况\s*[|｜]|关于人民网\s*[|｜]|"
    r"(?m:^[ \t]*(?:人民日报违法和不良信息举报电话|人民网服务邮箱)[:：])|"
    r"(?m:^[ \t]*互联网新闻信息服务许可证\s*\d{6,})|"
    r"(?m:^[ \t]*信息网络传播视听节目许可证\s*\d{6,})|"
    r"人\s*民\s*网\s*(?:股\s*份\s*有\s*限\s*公\s*司\s*)?版\s*权\s*所\s*有|"
    r"Copyright[^\n]{0,100}www\.people\.com\.cn[^\n]*all rights reserved",
    re.IGNORECASE,
)
# Other sites: match specific footer wording, not every mention of 版权/举报.
SITE_FOOTER_RE = re.compile(
    PEOPLE_FOOTER_RE.pattern + r"|"
    r"京ICP备10003349号(?:-\d+)?[^\n]{0,80}(?:央视网|中央广播电视总台)|"
    r"(?:中央广播电视总台\s*)?央视网\s*版权所有|"
    r"(?:中国新闻网|中新网|中国新闻社)\s*版权所有|"
    r"Copyright[^\n]{0,150}(?:cctv\.com|chinanews\.com(?:\.cn)?)[^\n]*|"
    r"(?m:^[ \t]*(?:关于我们|关于央视网|关于中新网)\s*[|｜][^\n]*"
    r"(?:联系我们|广告服务|网站地图|版权声明))",
    re.IGNORECASE,
)

# Output columns include decisions and their reasons.
FIELDS = [
    "article_id", "media", "year", "date", "title", "text", "url",
    "search_keywords", "matched_keywords", "source_file", "source_row",
    "decision", "reason", "duplicate_of", "duplicate_group",
    "strong_hits", "weak_context_hits", "false_positive_hits",
    "relevant_sentences", "relevant_ratio", "relevance_example",
    "entertainment_hits", "commercial_hits",
]


# 2. Find marriage-related words and sentences.
def relevance_evidence(title, text):
    strong = STRONG_KEYWORDS
    weak = WEAK_KEYWORDS

    combined = title + "\n" + text
    false_hits = [phrase for phrase in FALSE_POSITIVE_PATTERNS if phrase in combined]

    def mask_false_matches(value):
        # Ignore false matches here, but keep them in the saved text.
        for phrase in FALSE_POSITIVE_PATTERNS:
            value = value.replace(phrase, " " * len(phrase))
        return value

    # 青年 and 家庭 alone are not clear marriage context.
    anchors = [w for w in CONTEXT_KEYWORDS if w not in ("男女", "青年", "家庭")]

    def hits(sentence):
        strong_hits = [w for w in strong if w in sentence]
        weak_hits = []
        for word in weak:
            for match in re.finditer(re.escape(word), sentence):
                # Check 40 characters each way; a word cannot be its own context.
                nearby = sentence[max(0, match.start() - 40):match.end() + 40]
                nearby = nearby.replace(word, "")
                if any(anchor in nearby for anchor in anchors if anchor != word):
                    weak_hits.append(word)
                    break
        return strong_hits, weak_hits

    masked_title = mask_false_matches(title)
    title_strong, title_weak = hits(masked_title)
    sentences = [s.strip() for s in re.split(r"[。！？!?\n]+", mask_false_matches(text)) if s.strip()]
    relevant = []
    all_strong, all_weak = set(title_strong), set(title_weak)
    for sentence in sentences:
        strong_hits, weak_hits = hits(sentence)
        all_strong.update(strong_hits)
        all_weak.update(weak_hits)
        if strong_hits or weak_hits:
            relevant.append(sentence)
    return {
        "matched_keywords": "|".join(w for w in strong + weak if w in mask_false_matches(combined)),
        "strong_hits": "|".join(sorted(all_strong)),
        "weak_context_hits": "|".join(sorted(all_weak)),
        "false_positive_hits": "|".join(false_hits),
        "relevant_sentences": len(relevant),
        "relevant_ratio": round(len(relevant) / max(len(sentences), 1), 4),
        "relevance_example": (relevant[0] if relevant else masked_title if title_strong or title_weak else "")[:200],
    }, bool(title_strong), bool(title_weak)


# 3. Clean text, dates and links.
def normalize_text(value):
    value = re.sub(r"<[^>]+>", "", value or "")
    value = html.unescape(value).replace("\u3000", " ").replace("\xa0", " ")
    footer = SITE_FOOTER_RE.search(value)
    if footer:
        value = value[:footer.start()]
    lines = []
    for line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        if re.fullmatch(r"(?:责任编辑|编辑|责编)[:：].{0,40}", line):
            continue
        # Newspaper edition notes and trailing editor credits are not article text.
        if re.fullmatch(r"《\s*人民日报\s*》\s*[（(].*\d{4}年.*版\s*[）)]", line):
            continue
        line = re.sub(r"[（(【\[](?:责编|责任编辑|编辑)[:：][^）)】\]\n]{1,40}[）)】\]]\s*$", "", line).strip()
        # Remove standalone registration codes; retain news discussing regulation.
        if re.fullmatch(r"(?:京|沪|粤|津|浙|苏)?(?:ICP备|ICP证|公网安备)[A-Za-z0-9号\- ]+", line):
            continue
        if line in {"分享到", "分享", "返回顶部", "打印", "关闭", "推荐阅读", "相关阅读"}:
            continue
        # Remove long paragraphs repeated immediately by the page parser.
        if not line or (lines and len(line) >= 60 and line == lines[-1]):
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_date(value):
    match = re.match(r"^\s*(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", value or "")
    if match:
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            pass
    return ""


def normalize_url(value):
    parts = urlsplit((value or "").strip())
    host = (parts.hostname or "").lower()
    media = next((name for domain, name in SOURCES.items()
                  if host == domain or host.endswith("." + domain)), "")
    url = urlunsplit(("https", parts.netloc.lower(), parts.path, parts.query, ""))
    return url, media


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def comparison_text(value):
    # Ignore spaces and punctuation for comparison only; saved text stays intact.
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(c for c in value if c.isalnum())


# 4. Keep, exclude or send to manual review.
def screen_article(row, year):
    if not row["media"]:
        return "exclude", "outside_selected_sources"
    if not row["date"]:
        return "review", "missing_or_invalid_date"
    if row["date"][:4] != str(year):
        return "exclude", "outside_selected_year"
    if not row["text"]:
        return "exclude", "missing_text"
    if len(re.sub(r"\s", "", row["text"])) < MIN_TEXT_LENGTH:
        return "review", "short_or_incomplete_text"
    chinese = len(re.findall(r"[\u4e00-\u9fff]", row["text"]))
    if chinese / max(len(re.sub(r"\s", "", row["text"])), 1) < 0.3:
        return "review", "possibly_non_chinese_text"
    evidence, title_strong, title_weak = relevance_evidence(row["title"], row["text"])
    row.update(evidence)
    if not row["strong_hits"] and not row["weak_context_hits"]:
        return "exclude", "no_topic_evidence_or_weak_terms_without_context"
    if not row["title"]:
        return "review", "missing_title"
    combined = row["title"] + "\n" + row["text"]
    row["entertainment_hits"] = "|".join(w for w in ENTERTAINMENT_TERMS if w in combined)
    row["commercial_hits"] = "|".join(w for w in COMMERCIAL_TERMS if w in combined)
    entertainment_pattern = "|".join(re.escape(w) for w in sorted(ENTERTAINMENT_TERMS, key=len, reverse=True))
    entertainment_count = len(re.findall(entertainment_pattern, row["text"]))
    content_flags = []
    if any(w in row["title"] for w in ENTERTAINMENT_TERMS) or entertainment_count >= 3:
        content_flags.append("possible_entertainment_focus")
    if row["commercial_hits"]:
        content_flags.append("possible_marketing")
    if content_flags:
        return "review", "|".join(content_flags)
    # A strong title must also have support in the body.
    if title_strong and row["relevant_sentences"] >= 1:
        return "keep", "strong_title_with_body_support"
    if (row["relevant_sentences"] >= MIN_RELEVANT_SENTENCES
            and row["relevant_ratio"] >= MIN_RELEVANT_RATIO):
        return "keep", "sustained_topic_discussion"
    if title_weak:
        return "review", "weak_title_with_context_needs_review"
    return "review", "incidental_or_borderline_topic_discussion"


# 5. Read the selected year's files and screen each article.
def load_articles(paths, year):
    rows = []
    for path in paths:
        print(f"[Read] {path.name}", flush=True)
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"date", "title", "text", "url"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path.name}: missing columns {required - set(reader.fieldnames or [])}")
            for number, original in enumerate(reader, start=2):
                url, media = normalize_url(original["url"])
                title = normalize_text(original["title"])
                text = normalize_text(original["text"])
                article_date = normalize_date(original["date"])
                row = dict.fromkeys(FIELDS, "")
                row.update(article_id="A_" + digest(f"{path.name}:{number}:{url}"),
                           media=media or original.get("media", "unknown"),
                           year=article_date[:4] or "unknown", date=article_date,
                           title=title, text=text, url=url,
                           search_keywords=original.get("search_keywords", ""),
                           source_file=path.name, source_row=number)
                source_label = row["media"]
                row["media"] = media
                row["decision"], row["reason"] = screen_article(row, year)
                row["media"] = source_label
                rows.append(row)
    return rows


# 6. Remove same-source duplicates; label cross-media copies.
def remove_duplicates(rows):
    # Prefer a kept article, then the longer copy.
    rank = {"keep": 0, "review": 1, "exclude": 2}
    ordered = sorted(rows, key=lambda r: (rank[r["decision"]], -len(r["text"]), r["article_id"]))
    seen_urls, seen_texts = {}, {}
    seen_titles = defaultdict(list)
    groups = defaultdict(list)
    for row in ordered:
        if row["decision"] == "exclude":
            continue
        comparable = comparison_text(row["text"])
        fingerprint = digest(comparable) if comparable else ""
        same_text_key = (row["media"], fingerprint)
        duplicate = seen_urls.get(row["url"]) or (seen_texts.get(same_text_key) if fingerprint else None)
        if duplicate:
            row.update(decision="exclude", reason="duplicate_url_or_same_media_text",
                       duplicate_of=duplicate["article_id"])
            continue
        # Same media + same title + almost identical body: review, not auto-delete.
        title_key = (row["media"], comparison_text(row["title"]))
        if title_key[1] and len(comparable) >= 200:
            for old, old_text in seen_titles[title_key]:
                length_ratio = min(len(comparable), len(old_text)) / max(len(comparable), len(old_text), 1)
                if length_ratio >= 0.95 and SequenceMatcher(None, comparable, old_text, autojunk=False).ratio() >= 0.97:
                    row.update(decision="review", reason="possible_near_duplicate",
                               duplicate_of=old["article_id"])
                    break
        if title_key[1]:
            seen_titles[title_key].append((row, comparable))
        seen_urls[row["url"]] = row
        if fingerprint:
            seen_texts[same_text_key] = row
            groups[fingerprint].append(row)
    for fingerprint, copies in groups.items():
        if len({r["media"] for r in copies}) > 1:
            for row in copies:
                row["duplicate_group"] = "D_" + fingerprint


# 7. Save CSV files.
def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


# 8. Run the cleaning steps and save to a new folder.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2023, 2024, 2025], default=YEAR_TO_CLEAN)
    parser.add_argument("--inputs", nargs="+", type=Path, help="Optional explicit input CSV paths")
    parser.add_argument("--output-dir", type=Path, help="Must be a new folder; never overwrites earlier results")
    args = parser.parse_args()
    if args.year not in (2023, 2024, 2025):
        parser.error("YEAR_TO_CLEAN must be 2023, 2024 or 2025.")
    root = Path(__file__).resolve().parent
    # China News Service is already included in the merged annual file.
    paths = args.inputs or [root / f"official_media_articles_{args.year}.csv"]
    for path in paths:
        if not path.is_file():
            parser.error(f"Input CSV not found: {path}")
    output = args.output_dir or root / (f"cleaned_marriage_{args.year}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists():
        parser.error("Output folder already exists. Choose a new folder to avoid overwriting results.")
    print(f"[Settings] Year: {args.year} | Topic: marriage", flush=True)
    rows = load_articles(paths, args.year)
    remove_duplicates(rows)
    rows.sort(key=lambda r: (r["year"], r["media"], r["date"], r["article_id"]))
    output.mkdir(parents=True)
    # One file for relevant/review articles; one file for excluded articles.
    labels = {"keep": "高度相关（规则初筛）", "review": "需要人工检查", "exclude": "排除"}
    for row in rows:
        row["相关性标注"] = labels[row["decision"]]
        if row["reason"] == "possible_near_duplicate":
            row["相关性标注"] = "需要人工检查（疑似重复）"
        elif row["duplicate_group"]:
            row["相关性标注"] += "；跨媒体转载"
    output_fields = ["media", "date", "title", "text", "url", "matched_keywords", "相关性标注"]
    write_csv(output / f"marriage_articles_{args.year}_clean.csv", output_fields,
              [r for r in rows if r["decision"] in ("keep", "review")])
    write_csv(output / f"marriage_articles_{args.year}_excluded.csv", output_fields + ["reason"],
              [r for r in rows if r["decision"] == "exclude"])
    print("[Done]", dict(Counter(r["decision"] for r in rows)), flush=True)
    print(f"Results: {output}")
    print("Check manual-review cases and sample both keep/exclude before analysis.")
    print("Dates come from the input CSV; publication/update date accuracy needs verification.")


if __name__ == "__main__":
    main()
