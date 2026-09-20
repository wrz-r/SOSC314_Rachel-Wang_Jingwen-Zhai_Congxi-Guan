"""Preprocess Chinese official-media articles for bag-of-words and LDA.

1. tokens_basic: four public Chinese stopword lists are merged.
2. tokens_expanded: the merged list plus news boilerplate is used.

Each row remains one article.  The title and article text are combined before
tokenization, while the original title and text are retained for checking.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from pathlib import Path


# =========================================================
# 1. FILE SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "official_media_articles_2021_2025_final_revised.csv"
DEFAULT_OUTPUT = BASE_DIR / "official_media_articles_2021_2025_preprocessed.csv"
STOPWORD_DIR = BASE_DIR / "stopwords"


# =========================================================
# 2. DOMAIN PHRASES
# Keep these expressions as complete words during segmentation.
# =========================================================

DOMAIN_PHRASES = [
    "婚姻登记",
    "结婚登记",
    "离婚登记",
    "离婚冷静期",
    "高价彩礼",
    "天价彩礼",
    "零彩礼",
    "婚俗改革",
    "移风易俗",
    "跨省通办",
    "全国通办",
    "婚恋平台",
    "婚恋交友",
    "家庭暴力",
    "夫妻共同财产",
    "夫妻共同债务",
    "结婚意愿",
    "初婚年龄",
    "婚姻家庭纠纷",
    "人身安全保护令",
]


# =========================================================
# 3. STOPWORD DICTIONARIES
# The basic version merges four public Chinese stopword lists.
# Source collection: https://github.com/goto456/stopwords
# =========================================================

STOPWORD_FILES = [
    STOPWORD_DIR / "hit_stopwords.txt",
    STOPWORD_DIR / "baidu_stopwords.txt",
    STOPWORD_DIR / "scu_stopwords.txt",
    STOPWORD_DIR / "cn_stopwords.txt",
]

# These words may carry substantive meaning in marriage discourse.
# They are retained even if a public stopword list contains them.
PROTECTED_TERMS = {
    # Negation and attitude
    "不", "没", "没有", "未", "无", "不是", "不想", "不愿", "不敢", "不能",

    # Marriage-related concepts
    "婚姻", "结婚", "离婚", "婚恋", "恋爱", "对象", "伴侣", "相亲",
    "单身", "不婚", "晚婚", "恐婚", "催婚", "彩礼", "婚俗", "家庭",
    "夫妻", "配偶", "情侣", "青年", "年轻人", "女性", "男性",

    # Policy and legal concepts
    "政策", "登记", "法律", "法院", "民政",
}

# The expanded version adds corpus-specific words common in news writing.
NEWS_BOILERPLATE_STOPWORDS = {
    # News production
    "记者", "编辑", "责编", "责任编辑", "通讯员", "摄影", "摄制", "报道",
    "新闻记者", "媒体报道", "新闻报道",
    "消息", "发布", "表示", "指出", "介绍", "认为", "强调", "透露", "获悉",
    "了解到", "据了解", "快讯", "日电", "本报讯", "现场",

    # Source and platform names
    "人民网", "人民日报", "央视网", "央视", "中央电视台", "中新网", "中新社",
    "中国新闻网", "新华网", "新华社", "中国日报", "光明网", "光明日报",
    "中国青年网", "本报", "本报记者", "本网", "通讯社", "新闻网", "客户端",
    "微信公众号", "官网", "来源", "作者", "原标题",

    # Copyright notices and web-page boilerplate
    "版权", "版权所有", "版权声明", "转载", "禁止转载", "转载请注明",
    "未经授权", "二维码", "扫一扫", "稿件", "投稿", "投稿邮箱",
    "阅读原文", "点击查看", "点击进入", "下载客户端", "返回首页",
    "网站", "电话", "注册",

    # Relative time and generic news wording
    "日前", "近日", "当天", "今日", "昨日", "上午", "下午", "中午", "早上",
    "晚间", "目前", "今年", "去年", "近年来", "此次", "相关", "进行", "通过",
    "进一步", "工作",
}

# Jieba sometimes splits boilerplate phrases such as "未经授权" into two
# ordinary words. Keeping these phrases intact allows the expanded dictionary
# to remove the boilerplate without deleting broad words such as "授权".
NEWS_BOILERPLATE_PHRASES = {
    "新闻记者", "媒体报道", "新闻报道",
    "版权所有", "版权声明", "禁止转载", "转载请注明", "未经授权",
    "投稿邮箱", "阅读原文", "点击查看", "点击进入", "下载客户端",
    "返回首页", "扫一扫",
}

# One-character words are usually noisy. These substantive words are retained.
KEEP_SINGLE_CHAR = {"不", "没", "未", "无", "婚", "嫁", "娶"}


# =========================================================
# 4. TEXT CLEANING AND TOKENIZATION
# =========================================================

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
SPACE_RE = re.compile(r"\s+")
VALID_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9.%]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def normalize_text(value: object) -> str:
    """Remove markup and normalize whitespace without changing meaning."""
    text = html.unescape(str(value or ""))
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u3000", " ").replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def load_basic_stopwords() -> set[str]:
    """Merge the four public lists and restore protected research terms."""
    missing_files = [path for path in STOPWORD_FILES if not path.exists()]
    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise SystemExit(f"Missing stopword file(s):\n{missing_text}")

    combined: set[str] = set()
    for path in STOPWORD_FILES:
        with path.open("r", encoding="utf-8-sig") as file:
            combined.update(
                line.strip().lower()
                for line in file
                if line.strip()
            )

    protected = PROTECTED_TERMS | set(DOMAIN_PHRASES) | KEEP_SINGLE_CHAR
    protected_hits = combined & protected
    basic_stopwords = combined - protected

    print(
        f"Loaded {len(combined):,} unique stopwords from four public lists; "
        f"retained {len(protected_hits)} protected term(s)."
    )
    return basic_stopwords


def load_jieba():
    """Load jieba and give a clear instruction if it is not installed."""
    try:
        import jieba
    except ImportError as exc:
        raise SystemExit(
            "Missing package: jieba\n"
            "Install it with: python3 -m pip install jieba"
        ) from exc

    for phrase in DOMAIN_PHRASES + sorted(NEWS_BOILERPLATE_PHRASES):
        jieba.add_word(phrase, freq=100000)
    return jieba


def tokenize(text: str, jieba_module, stopwords: set[str]) -> list[str]:
    """Segment Chinese text and remove punctuation, noise, and stopwords."""
    tokens: list[str] = []

    for raw_token in jieba_module.lcut(text, cut_all=False):
        token = raw_token.strip().lower()

        if not token or token in stopwords:
            continue
        if not VALID_TOKEN_RE.fullmatch(token):
            continue
        if len(token) == 1 and token not in KEEP_SINGLE_CHAR and not token.isdigit():
            continue

        # Numbers are retained because values such as years, ages, and
        # bride-price amounts may be substantively meaningful.
        if NUMBER_RE.fullmatch(token) or len(token) >= 2 or token in KEEP_SINGLE_CHAR:
            tokens.append(token)

    return tokens


# =========================================================
# 5. PREPROCESS THE CSV
# =========================================================

def preprocess(input_path: Path, output_path: Path, min_tokens: int) -> None:
    jieba = load_jieba()
    basic_stopwords = load_basic_stopwords()
    expanded_stopwords = (
        basic_stopwords | NEWS_BOILERPLATE_STOPWORDS
    ) - PROTECTED_TERMS

    csv.field_size_limit(sys.maxsize)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0
    blank_text = 0
    too_short = 0
    basic_token_total = 0
    expanded_token_total = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"media", "date", "title", "text", "url", "matched_keywords"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        original_fields = list(reader.fieldnames or [])
        extra_fields = [
            "year",
            "tokens_basic",
            "tokens_expanded",
            "token_count_basic",
            "token_count_expanded",
        ]

        with output_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=original_fields + extra_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                title = normalize_text(row.get("title", ""))
                article_text = normalize_text(row.get("text", ""))

                # A title alone is not enough for article-level topic modeling.
                if not article_text:
                    blank_text += 1
                    continue

                document = f"{title}。{article_text}" if title else article_text
                basic_tokens = tokenize(document, jieba, basic_stopwords)
                expanded_tokens = tokenize(document, jieba, expanded_stopwords)

                if len(expanded_tokens) < min_tokens:
                    too_short += 1
                    continue

                row["title"] = title
                row["text"] = article_text
                row["year"] = str(row.get("date", ""))[:4]
                row["tokens_basic"] = " ".join(basic_tokens)
                row["tokens_expanded"] = " ".join(expanded_tokens)
                row["token_count_basic"] = len(basic_tokens)
                row["token_count_expanded"] = len(expanded_tokens)
                writer.writerow(row)

                saved += 1
                basic_token_total += len(basic_tokens)
                expanded_token_total += len(expanded_tokens)

                if total % 250 == 0:
                    print(f"Processed {total:,} rows; retained {saved:,}", flush=True)

    print("\nPreprocessing complete")
    print(f"Input rows: {total:,}")
    print(f"Saved rows: {saved:,}")
    print(f"Skipped blank-text rows: {blank_text:,}")
    print(f"Skipped rows with fewer than {min_tokens} tokens: {too_short:,}")
    if saved:
        print(f"Average basic tokens: {basic_token_total / saved:.1f}")
        print(f"Average expanded tokens: {expanded_token_total / saved:.1f}")
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess official-media marriage articles for BOW and LDA."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=10,
        help="Minimum expanded-token count required to retain an article (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.min_tokens < 1:
        raise SystemExit("--min-tokens must be at least 1")
    preprocess(args.input, args.output, args.min_tokens)


if __name__ == "__main__":
    main()
