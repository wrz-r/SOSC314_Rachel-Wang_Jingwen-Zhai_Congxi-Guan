# -*- coding: utf-8 -*-
"""Preprocess user-initiated Weibo posts for topic modeling."""

import argparse
import csv
import html
import re
import sys
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "2021-2025_eligible_m_c_r_h_dedup copy.csv"
DEFAULT_OUTPUT = BASE_DIR / "weibo_user_posts_2021_2025_preprocessed.csv"
STOPWORD_DIR = BASE_DIR / "stopwords"

STOPWORD_FILES = [
    STOPWORD_DIR / "hit_stopwords.txt",
    STOPWORD_DIR / "baidu_stopwords.txt",
    STOPWORD_DIR / "scu_stopwords.txt",
    STOPWORD_DIR / "cn_stopwords.txt",
]

# Keep important marriage expressions as complete words.
DOMAIN_PHRASES = [
    "婚姻登记", "结婚登记", "离婚登记", "离婚冷静期",
    "高价彩礼", "天价彩礼", "零彩礼", "婚俗改革", "移风易俗",
    "婚恋平台", "婚恋交友", "家庭暴力", "夫妻共同财产",
    "夫妻共同债务", "结婚意愿", "婚恋观", "恋爱观", "择偶标准",
    "不婚主义", "恐婚恐育", "催婚催育", "跨省通办", "全国通办",
    "不想结婚", "不愿结婚", "不敢结婚", "不想恋爱", "不想相亲",
]

# These words are important for marriage attitudes and must not be removed.
PROTECTED_TERMS = {
    "婚姻", "结婚", "离婚", "婚恋", "恋爱", "对象", "伴侣", "相亲",
    "单身", "不婚", "晚婚", "恐婚", "催婚", "彩礼", "婚俗", "家庭",
    "夫妻", "配偶", "情侣", "青年", "年轻人", "女性", "男性", "父母",
    "压力", "自由", "幸福", "焦虑", "害怕", "愿意", "拒绝", "支持",
}

# Platform words that do not describe the substantive topic.
WEIBO_STOPWORDS = {
    "微博", "博文", "博主", "网友", "评论区", "热搜", "超话", "话题",
    "视频", "微博视频", "秒拍视频", "网页链接", "客户端", "转发微博",
    "转发", "评论", "点赞", "关注", "粉丝", "私信", "链接", "全文",
    "展开全文", "图片", "长图", "来源", "原标题", "记者", "编辑", "报道",
    "发布", "表示", "指出", "介绍", "获悉", "本报讯", "日电",
    # Very common conversational words that blur topic distinctions
    "不", "没", "没有", "不是", "真的", "感觉", "觉得", "其实", "可能",
    "就是", "还是", "已经", "现在", "这样", "这种", "一个", "自己",
    "生活", "朋友", "喜欢",
}

KEEP_SINGLE_CHAR = {"婚", "嫁", "娶"}


# ============================================================
# TEXT CLEANING
# ============================================================

HASHTAG_RE = re.compile(r"#([^#]+)#")
MENTION_RE = re.compile(r"@[0-9A-Za-z_\-\u4e00-\u9fff]{1,30}")
URL_RE = re.compile(r"https?://\S+|www\.\S+|t\.cn/\S+", re.I)
VIDEO_RE = re.compile(r"L[0-9A-Za-z_\-\u4e00-\u9fff]+的微博视频")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
VALID_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9.%]+")


def clean_weibo_text(value: str) -> str:
    """Remove platform formatting while keeping hashtag content."""
    text = html.unescape(str(value or ""))
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = MENTION_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = VIDEO_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = re.sub(r"(?:O网页链接|网页链接|展开全文|微博视频|秒拍视频)", " ", text)
    text = text.replace("\u3000", " ").replace("\x00", " ")
    return SPACE_RE.sub(" ", text).strip()


def load_stopwords() -> set[str]:
    """Merge four public Chinese stopword lists and Weibo-specific words."""
    stopwords = set()
    for path in STOPWORD_FILES:
        if not path.exists():
            raise SystemExit(f"Missing stopword file: {path}")
        with path.open("r", encoding="utf-8-sig") as file:
            stopwords.update(line.strip().lower() for line in file if line.strip())

    protected = PROTECTED_TERMS | set(DOMAIN_PHRASES) | KEEP_SINGLE_CHAR
    return (stopwords | WEIBO_STOPWORDS) - protected


def load_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise SystemExit("Install jieba with: python3 -m pip install jieba") from exc

    for phrase in DOMAIN_PHRASES:
        jieba.add_word(phrase, freq=100000)
    return jieba


def tokenize(text: str, jieba, stopwords: set[str]) -> list[str]:
    tokens = []
    for raw_token in jieba.lcut(text, cut_all=False):
        token = raw_token.strip().lower()
        if not token or token in stopwords:
            continue
        if not VALID_TOKEN_RE.fullmatch(token):
            continue
        if len(token) == 1 and token not in KEEP_SINGLE_CHAR and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


# ============================================================
# PREPROCESS CSV
# ============================================================

def preprocess(input_file: Path, output_file: Path, min_tokens: int) -> None:
    jieba = load_jieba()
    stopwords = load_stopwords()
    seen_texts = set()

    csv.field_size_limit(sys.maxsize)
    total = saved = blank = duplicate = too_short = 0

    with input_file.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"sampling_year", "query_keyword", "id", "user_id", "微博正文", "发布时间"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

        output_fields = list(reader.fieldnames or []) + [
            "year", "clean_text", "tokens_expanded", "token_count_expanded"
        ]

        with output_file.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                total += 1
                cleaned = clean_weibo_text(row.get("微博正文", ""))
                if not cleaned:
                    blank += 1
                    continue

                # Deduplicate by the complete cleaned text, not by damaged IDs.
                text_key = re.sub(r"\s+", "", cleaned).lower()
                if text_key in seen_texts:
                    duplicate += 1
                    continue
                seen_texts.add(text_key)

                tokens = tokenize(cleaned, jieba, stopwords)
                if len(tokens) < min_tokens:
                    too_short += 1
                    continue

                year = str(row.get("sampling_year", "")).strip()
                if not re.fullmatch(r"20\d{2}", year):
                    year_match = re.search(r"20\d{2}", row.get("发布时间", ""))
                    year = year_match.group() if year_match else ""

                row["year"] = year
                row["clean_text"] = cleaned
                row["tokens_expanded"] = " ".join(tokens)
                row["token_count_expanded"] = len(tokens)
                writer.writerow(row)
                saved += 1

                if total % 5000 == 0:
                    print(f"Processed {total:,}; retained {saved:,}", flush=True)

    print("\nPreprocessing complete")
    print(f"Input rows: {total:,}")
    print(f"Saved rows: {saved:,}")
    print(f"Blank rows removed: {blank:,}")
    print(f"Exact duplicate texts removed: {duplicate:,}")
    print(f"Posts with fewer than {min_tokens} tokens removed: {too_short:,}")
    print(f"Output: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Weibo posts for NMF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-tokens", type=int, default=5)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    preprocess(args.input, args.output, args.min_tokens)


if __name__ == "__main__":
    main()
