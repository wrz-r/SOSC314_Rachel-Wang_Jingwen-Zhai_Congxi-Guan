import pandas as pd
import re
import html
import sys
from difflib import SequenceMatcher


# ============================================================
# 0. YEAR
# ============================================================

YEAR = 2021

INPUT_FILE = f"official_media_articles_{YEAR}_marriage_final.csv"

OUTPUT_FILE = (
    f"official_media_articles_{YEAR}_marriage_final_cleaned.csv"
)

REVIEW_FILE = (
    f"official_media_articles_{YEAR}_marriage_manual_review.csv"
)

DUPLICATE_FILE = (
    f"official_media_articles_{YEAR}_marriage_duplicates.csv"
)


# ============================================================
# 1. MARRIAGE KEYWORDS
# ============================================================

MARRIAGE_KEYWORDS = [
    "婚姻",
    "结婚",
    "婚恋",
    "恋爱",
    "对象",
    "伴侣",
    "相亲",
    "单身",
    "不婚",
    "晚婚",
    "恐婚",
    "催婚",
    "婚姻登记",
    "彩礼",
    "离婚冷静期",
]


# ============================================================
# 2. NON-ROMANTIC OBJECT PATTERNS
# ============================================================

NON_ROMANTIC_OBJECT_PATTERNS = [
    "研究对象",
    "调查对象",
    "服务对象",
    "帮扶对象",
    "救助对象",
    "管理对象",
    "监管对象",
    "执法对象",
    "保护对象",
    "援助对象",
    "资助对象",
    "扶持对象",
    "培训对象",
    "教育对象",
    "消费者对象",
    "采访对象",
    "采访的对象",
    "受访对象",
    "犯罪对象",
    "攻击对象",
    "制裁对象",
    "打击对象",
    "目标对象",
    "对象国",
    "对象企业",
    "对象地区",
    "对象群体",
    "对象人群",
]


# ============================================================
# 3. HTML / WEBSITE CLEANING
# ============================================================

def clean_html(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove script/style blocks
    text = re.sub(
        r"<script.*?>.*?</script>",
        " ",
        text,
        flags=re.I | re.S
    )

    text = re.sub(
        r"<style.*?>.*?</style>",
        " ",
        text,
        flags=re.I | re.S
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. REMOVE WEBSITE FOOTERS / COPYRIGHT
# ============================================================

FOOTER_PATTERNS = [

    r"版权所有.{0,100}",
    r"版权归.{0,100}",
    r"未经许可.{0,100}",
    r"未经授权.{0,100}",
    r"转载请注明.{0,100}",
    r"转载请保留.{0,100}",
    r"本文来源.{0,100}",
    r"来源[:：].{0,80}",
    r"责任编辑[:：].{0,50}",
    r"编辑[:：].{0,50}",
    r"记者[:：].{0,50}",
    r"审校[:：].{0,50}",
    r"责编[:：].{0,50}",
    r"校对[:：].{0,50}",
    r"策划[:：].{0,50}",
    r"制作[:：].{0,50}",
    r"视觉[:：].{0,50}",
    r"图片来源[:：].{0,80}",
    r"图片由.{0,80}",
    r"资料来源[:：].{0,80}",
    r"本网站.{0,100}",
    r"本网页.{0,100}",
    r"本平台.{0,100}",
    r"免责声明.{0,200}",
]


def remove_footer(text):

    if not text:
        return ""

    for pattern in FOOTER_PATTERNS:

        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.I
        )

    # Normalize whitespace again
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 5. NORMALIZE TEXT FOR DUPLICATE DETECTION
# ============================================================

def normalize_for_duplicate(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove whitespace
    text = re.sub(
        r"\s+",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]《》<>「」\-—_]",
        "",
        text
    )

    return text.lower()


def normalize_title(title):

    if pd.isna(title):
        return ""

    title = str(title)

    title = re.sub(
        r"\s+",
        "",
        title
    )

    title = re.sub(
        r"[，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]《》<>「」\-—_]",
        "",
        title
    )

    return title.lower()


# ============================================================
# 6. CHECK BASIC DATA QUALITY
# ============================================================

def check_basic_quality(row):

    media = str(row["media"]).strip()
    date = str(row["date"]).strip()
    text = str(row["text"]).strip()

    if not media:
        return "missing_media"

    if not date:
        return "missing_date"

    if not text:
        return "missing_text"

    if len(text) < 50:
        return "text_too_short"

    return "valid"


# ============================================================
# 7. OBJECT CONTEXT
# ============================================================

def classify_object_context(text):

    if "对象" not in text:
        return "no_object"

    for pattern in NON_ROMANTIC_OBJECT_PATTERNS:

        if pattern in text:
            return "non_romantic"

    romantic_patterns = [

        "找对象",
        "介绍对象",
        "相亲对象",
        "结婚对象",
        "恋爱对象",
        "没有对象",
        "有对象",
        "谈对象",
        "处对象",
        "对象父母",
        "对象家人",
        "对象家长",
        "和对象",
        "跟对象",
        "与对象",
        "自己的对象",
        "另一半",
    ]

    for pattern in romantic_patterns:

        if pattern in text:
            return "romantic"

    return "ambiguous"


# ============================================================
# 8. MARRIAGE RELEVANCE
# ============================================================

def calculate_relevance(row):

    title = str(row["title"])
    text = str(row["text"])

    combined = title + " " + text

    matched = []

    for keyword in MARRIAGE_KEYWORDS:

        if keyword in combined:

            # Special treatment for "对象"
            if keyword == "对象":

                object_class = classify_object_context(combined)

                if object_class == "non_romantic":
                    continue

                if object_class == "ambiguous":
                    continue

            matched.append(keyword)

    # Keyword in title is stronger evidence
    title_matches = []

    for keyword in MARRIAGE_KEYWORDS:

        if keyword in title:

            if keyword == "对象":

                object_class = classify_object_context(combined)

                if object_class == "non_romantic":
                    continue

                if object_class == "ambiguous":
                    continue

            title_matches.append(keyword)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if len(title_matches) >= 1:

        return (
            "highly_relevant",
            ";".join(matched),
            ";".join(title_matches)
        )

    if len(matched) >= 2:

        return (
            "highly_relevant",
            ";".join(matched),
            ""
        )

    if len(matched) == 1:

        return (
            "manual_review",
            ";".join(matched),
            ""
        )

    return (
        "not_relevant",
        "",
        ""
    )


# ============================================================
# 9. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print(f"LOADING {YEAR} DATA")
print("=" * 70)

try:

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

except FileNotFoundError:

    print(f"\nERROR: Cannot find {INPUT_FILE}")

    print("\nCurrent directory should contain:")
    print(INPUT_FILE)

    sys.exit(1)


print(f"Input file: {INPUT_FILE}")
print(f"Rows: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 10. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "media",
    "date",
    "title",
    "text",
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 11. BASIC CLEANING
# ============================================================

print("\n" + "=" * 70)
print("CLEANING TEXT")
print("=" * 70)

df["media"] = (
    df["media"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["date"] = (
    df["date"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["text"] = df["text"].fillna("")


# Clean HTML
df["text"] = df["text"].apply(clean_html)

# Remove website footer / copyright
df["text"] = df["text"].apply(remove_footer)

# Normalize title
df["title"] = df["title"].apply(clean_html)


# ============================================================
# 12. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATA QUALITY")
print("=" * 70)

df["data_quality"] = df.apply(
    check_basic_quality,
    axis=1
)

print(
    df["data_quality"]
    .value_counts()
)


# ============================================================
# 13. MARRIAGE RELEVANCE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING MARRIAGE RELEVANCE")
print("=" * 70)

results = df.apply(
    calculate_relevance,
    axis=1
)

df["relevance"] = results.apply(
    lambda x: x[0]
)

df["marriage_keywords"] = results.apply(
    lambda x: x[1]
)

df["title_keywords"] = results.apply(
    lambda x: x[2]
)


print(
    df["relevance"]
    .value_counts()
)


# ============================================================
# 14. OBJECT REVIEW FLAG
# ============================================================

df["object_context"] = df["text"].apply(
    classify_object_context
)


# ============================================================
# 15. REMOVE CLEARLY INVALID ARTICLES
# ============================================================

before = len(df)

df = df[
    df["data_quality"] == "valid"
].copy()

df = df[
    df["relevance"] != "not_relevant"
].copy()

after = len(df)

print("\n" + "=" * 70)
print("REMOVE CLEARLY INVALID ARTICLES")
print("=" * 70)

print(f"Before: {before}")
print(f"After:  {after}")
print(f"Removed: {before - after}")


# ============================================================
# 16. CREATE DUPLICATE KEYS
# ============================================================

df["normalized_title"] = df["title"].apply(
    normalize_title
)

df["normalized_text"] = df["text"].apply(
    normalize_for_duplicate
)


# ============================================================
# 17. SAME-MEDIA DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING SAME-MEDIA DUPLICATES")
print("=" * 70)

df["same_media_duplicate"] = False

duplicate_groups = []

for media, group in df.groupby("media"):

    # Exact normalized title duplicates
    title_counts = (
        group["normalized_title"]
        .value_counts()
    )

    duplicate_titles = title_counts[
        title_counts > 1
    ].index

    for title in duplicate_titles:

        if not title:
            continue

        indices = group[
            group["normalized_title"] == title
        ].index.tolist()

        if len(indices) > 1:

            for idx in indices:
                df.loc[
                    idx,
                    "same_media_duplicate"
                ] = True

            duplicate_groups.append({
                "type": "same_media",
                "media": media,
                "indices": indices,
                "reason": "same_normalized_title"
            })


# ============================================================
# 18. EXACT TEXT DUPLICATES WITHIN MEDIA
# ============================================================

for media, group in df.groupby("media"):

    text_counts = (
        group["normalized_text"]
        .value_counts()
    )

    duplicate_texts = text_counts[
        text_counts > 1
    ].index

    for text_key in duplicate_texts:

        if not text_key:
            continue

        indices = group[
            group["normalized_text"] == text_key
        ].index.tolist()

        if len(indices) > 1:

            for idx in indices:
                df.loc[
                    idx,
                    "same_media_duplicate"
                ] = True

            duplicate_groups.append({
                "type": "same_media",
                "media": media,
                "indices": indices,
                "reason": "same_normalized_text"
            })


print(
    "Same-media duplicate rows:",
    df["same_media_duplicate"].sum()
)


# ============================================================
# 19. CROSS-MEDIA REPRINT DETECTION
# ============================================================

print("\n" + "=" * 70)
print("CHECKING CROSS-MEDIA REPRINTS")
print("=" * 70)

df["cross_media_reprint"] = False
df["cross_media_match_type"] = ""


# ------------------------------------------------------------
# 19A. Same normalized title across different media
# ------------------------------------------------------------

title_groups = (
    df[
        df["normalized_title"] != ""
    ]
    .groupby("normalized_title")
)


for title, group in title_groups:

    media_count = group["media"].nunique()

    if media_count > 1:

        for idx in group.index:

            df.loc[
                idx,
                "cross_media_reprint"
            ] = True

            df.loc[
                idx,
                "cross_media_match_type"
            ] = "same_title"


# ------------------------------------------------------------
# 19B. Exact normalized text across different media
# ------------------------------------------------------------

text_groups = (
    df[
        df["normalized_text"] != ""
    ]
    .groupby("normalized_text")
)


for text_key, group in text_groups:

    media_count = group["media"].nunique()

    if media_count > 1:

        for idx in group.index:

            df.loc[
                idx,
                "cross_media_reprint"
            ] = True

            df.loc[
                idx,
                "cross_media_match_type"
            ] = "same_text"


print(
    "Cross-media reprint rows:",
    df["cross_media_reprint"].sum()
)


# ============================================================
# 20. MANUAL REVIEW FILE
# ============================================================

manual_review = df[
    (
        df["relevance"] == "manual_review"
    )
    |
    (
        df["object_context"] == "ambiguous"
    )
    |
    (
        df["cross_media_reprint"]
    )
].copy()


manual_review.to_csv(
    REVIEW_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 70)
print("MANUAL REVIEW")
print("=" * 70)

print(
    f"Manual-review rows: {len(manual_review)}"
)

print(
    f"Saved to: {REVIEW_FILE}"
)


# ============================================================
# 21. DUPLICATE REPORT
# ============================================================

duplicates = df[
    (
        df["same_media_duplicate"]
    )
    |
    (
        df["cross_media_reprint"]
    )
].copy()


duplicates.to_csv(
    DUPLICATE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 70)
print("DUPLICATE REPORT")
print("=" * 70)

print(
    f"Duplicate / reprint rows: {len(duplicates)}"
)

print(
    f"Saved to: {DUPLICATE_FILE}"
)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

# Keep temporary normalization fields out of final corpus
df = df.drop(
    columns=[
        "normalized_title",
        "normalized_text",
    ],
    errors="ignore"
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING COMPLETE")
print("=" * 70)

print(
    f"Original rows:       {before}"
)

print(
    f"Final rows:          {len(df)}"
)

print(
    f"Removed rows:        {before - len(df)}"
)

print(
    f"Highly relevant:     "
    f"{(df['relevance'] == 'highly_relevant').sum()}"
)

print(
    f"Manual review:       "
    f"{(df['relevance'] == 'manual_review').sum()}"
)

print(
    f"Same-media duplicate:"
    f" {df['same_media_duplicate'].sum()}"
)

print(
    f"Cross-media reprint: "
    f"{df['cross_media_reprint'].sum()}"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nManual review:")
print(REVIEW_FILE)

print("\nDuplicate report:")
print(DUPLICATE_FILE)

print("\nDone.")