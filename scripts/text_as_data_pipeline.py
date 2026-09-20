import os
import re
from collections import Counter

import pandas as pd
import jieba
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ============================================================
# 0. CHINESE FONT SETUP
# ============================================================

# matplotlib can only use fonts registered in its font manager.
# Pick the first installed font from the candidates below and
# put it at the front of font.sans-serif, so every text element
# renders Chinese while Western text still falls back cleanly.

def setup_chinese_font():

    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Arial Unicode MS",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
    ]

    available = {
        f.name for f in fm.fontManager.ttflist
    }

    for name in candidates:

        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name] + [
                f
                for f in plt.rcParams["font.sans-serif"]
                if f != name
            ]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"Matplotlib Chinese font: {name}")
            return name

    print(
        "WARNING: no Chinese font found - "
        "CJK text may render as boxes"
    )
    return None


CHINESE_FONT = setup_chinese_font()


# ============================================================
# 1. SETTINGS
# ============================================================

# Change this to your actual CSV filename
INPUT_FILE = "official_media_articles_cleaned.csv"

# Folder where all results will be saved
OUTPUT_DIR = "text_as_data_output"

# Name of the text column in your CSV
TEXT_COLUMN = "text"


# ============================================================
# 2. MARRIAGE DICTIONARY
# ============================================================

MARRIAGE_WORDS = [
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
    "离婚冷静期"
]

# English glosses shown next to Chinese words in the figures
WORD_TRANSLATIONS = {
    "婚姻": "marriage",
    "结婚": "getting married",
    "婚恋": "marriage & dating",
    "恋爱": "romantic relationships",
    "对象": "romantic partner",
    "伴侣": "partner",
    "相亲": "matchmaking",
    "单身": "singlehood",
    "不婚": "non-marriage",
    "晚婚": "late marriage",
    "恐婚": "fear of marriage",
    "催婚": "marriage pressure",
    "婚姻登记": "marriage registration",
    "彩礼": "bride price",
    "离婚冷静期": "divorce cooling-off period",
    "家庭": "family",
    "青年": "youth",
    "年轻人": "young people",
    "生育": "fertility",
    "孩子": "children",
    "政策": "policy",
    "服务": "services",
    "登记": "registration",
    "许可证": "license",
    "观念": "attitudes / views",
    "发展": "development",
    "工作": "work",
    "社会": "society",
    "生活": "life",
    "问题": "issues",
    "建设": "construction",
    "文化": "culture",
    "活动": "activities",
    "中国": "China",
    "信息": "information",
    "网络": "internet",
    "经营": "business operations",
    "文明": "civility",
    "邮箱": "email",
    "办理": "administrative processing",
}


def bilingual_label(word):
    """Chinese word + English gloss in parentheses.

    If no translation exists, or the word is already Latin script
    (e.g. 'cn', 'people'), return the word unchanged so we never
    produce duplicates like '许可证 (许可证)'.
    """
    if word in WORD_TRANSLATIONS:
        return f"{word} ({WORD_TRANSLATIONS[word]})"
    return word


# ============================================================
# 3. CHINESE STOPWORDS
# ============================================================

# Basic Chinese stopword list.
# You can replace/expand this later with a larger Chinese
# stopword dictionary.

STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "也", "有", "就",
    "都", "而", "及", "或", "一个", "一种", "这", "那",
    "这些", "那些", "我们", "你们", "他们", "她们",
    "它们", "自己", "可以", "没有", "不是", "因为",
    "所以", "如果", "但是", "并且", "以及", "对于",
    "关于", "通过", "进行", "成为", "作为", "其中",
    "已经", "将", "被", "把", "从", "到", "对", "中",
    "上", "下", "里", "内", "外", "前", "后",
    "很", "更", "最", "还", "又", "也", "都",
    "啊", "呀", "吗", "呢", "吧", "啊", "哦",
    "说", "表示", "指出", "认为", "介绍",
    "记者", "据了解"
}


# ============================================================
# 4. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} rows.")
print("\nColumns:")
print(df.columns.tolist())


# Check text column
if TEXT_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{TEXT_COLUMN}' not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# Replace missing text with empty strings
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)


# ============================================================
# 6. BASIC TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Basic cleaning for Chinese web text.

    Removes:
    - URLs
    - HTML tags
    - excessive whitespace
    - most punctuation/symbols
    """

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Keep Chinese characters, English letters and numbers
    text = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9]",
        " ",
        text
    )

    # Remove excessive spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning text...")

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)


# ============================================================
# 7. CHINESE TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Chinese word segmentation using jieba.
    """

    return list(jieba.cut(text))


print("Tokenizing Chinese text...")

df["tokens"] = df["clean_text"].apply(tokenize)


# ============================================================
# 8. REMOVE STOPWORDS
# ============================================================

def remove_stopwords(tokens):
    """
    Remove:
    - Chinese stopwords
    - punctuation-like tokens
    - single-character tokens
    """

    result = []

    for word in tokens:

        word = word.strip()

        # Skip empty tokens
        if not word:
            continue

        # Skip stopwords
        if word in STOPWORDS:
            continue

        # Skip single Chinese characters
        if len(word) == 1 and re.match(
            r"[\u4e00-\u9fff]",
            word
        ):
            continue

        # Skip tokens that contain no Chinese/English/numbers
        if not re.search(
            r"[\u4e00-\u9fffA-Za-z0-9]",
            word
        ):
            continue

        result.append(word)

    return result


print("Removing stopwords...")

df["tokens_clean"] = df["tokens"].apply(remove_stopwords)


# ============================================================
# 9. CREATE TOKENIZED TEXT
# ============================================================

# Join tokens back together with spaces.
# This format is convenient for sklearn.

df["processed_text"] = df["tokens_clean"].apply(
    lambda tokens: " ".join(tokens)
)


# ============================================================
# 10. MARRIAGE DICTIONARY LABEL
# ============================================================

def identify_marriage_related(tokens):
    """
    Simple dictionary-based classification.

    1 = contains at least one marriage-related term
    0 = does not contain marriage-related terms
    """

    return int(
        any(word in tokens for word in MARRIAGE_WORDS)
    )


print("Applying marriage dictionary...")

df["marriage_related"] = df["tokens"].apply(
    identify_marriage_related
)


# ============================================================
# 11. COUNT MARRIAGE-RELATED DOCUMENTS
# ============================================================

n_marriage = df["marriage_related"].sum()
n_total = len(df)

print("\n" + "=" * 70)
print("MARRIAGE DICTIONARY RESULTS")
print("=" * 70)

print(
    f"Marriage-related documents: "
    f"{n_marriage:,} / {n_total:,}"
)

if n_total > 0:
    print(
        f"Percentage: "
        f"{n_marriage / n_total * 100:.2f}%"
    )


# ============================================================
# 12. WORD FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("WORD FREQUENCY")
print("=" * 70)

all_words = []

for tokens in df["tokens_clean"]:
    all_words.extend(tokens)

word_counts = Counter(all_words)

frequency_df = pd.DataFrame(
    word_counts.most_common(),
    columns=["word", "frequency"]
)

print("\nTop 30 words:")
print(
    frequency_df.head(30).to_string(index=False)
)


# Save frequency table
frequency_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "word_frequency.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. BAG-OF-WORDS
# ============================================================

print("\n" + "=" * 70)
print("BAG-OF-WORDS")
print("=" * 70)

# Keep documents that have some text
bow_df = df[
    df["processed_text"].str.strip() != ""
].copy()

vectorizer = CountVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    min_df=2
)

X_bow = vectorizer.fit_transform(
    bow_df["processed_text"]
)

feature_names = vectorizer.get_feature_names_out()

print(
    f"Number of documents: {X_bow.shape[0]:,}"
)

print(
    f"Number of vocabulary terms: {X_bow.shape[1]:,}"
)


# ============================================================
# 14. GLOBAL BAG-OF-WORDS FREQUENCY
# ============================================================

bow_frequencies = X_bow.sum(axis=0).A1

bow_frequency_df = pd.DataFrame({
    "word": feature_names,
    "frequency": bow_frequencies
})

bow_frequency_df = bow_frequency_df.sort_values(
    "frequency",
    ascending=False
).reset_index(drop=True)

print("\nTop 30 Bag-of-Words terms:")
print(
    bow_frequency_df.head(30).to_string(index=False)
)


bow_frequency_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "bag_of_words_frequency.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF")
print("=" * 70)

tfidf_vectorizer = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    min_df=2
)

X_tfidf = tfidf_vectorizer.fit_transform(
    bow_df["processed_text"]
)

tfidf_features = (
    tfidf_vectorizer.get_feature_names_out()
)

tfidf_scores = X_tfidf.mean(axis=0).A1

tfidf_df = pd.DataFrame({
    "word": tfidf_features,
    "mean_tfidf": tfidf_scores
})

tfidf_df = tfidf_df.sort_values(
    "mean_tfidf",
    ascending=False
).reset_index(drop=True)

print("\nTop 30 TF-IDF terms:")
print(
    tfidf_df.head(30).to_string(index=False)
)


tfidf_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tfidf_terms.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. TOP WORDS FIGURE
# ============================================================

TOP_N = 20

plot_df = frequency_df.head(TOP_N).sort_values(
    "frequency"
)

plot_df["label"] = plot_df["word"].apply(bilingual_label)

plt.figure(figsize=(11, 9))

plt.barh(
    plot_df["label"],
    plot_df["frequency"],
    color="steelblue"
)

plt.xlabel("Frequency")
plt.ylabel("Word")
plt.title(
    "Top 20 Words in Official-Media Marriage Discourse"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top_20_words.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. TOP TF-IDF TERMS FIGURE
# ============================================================

plot_tfidf = tfidf_df.head(TOP_N).sort_values(
    "mean_tfidf"
)

plot_tfidf["label"] = plot_tfidf["word"].apply(bilingual_label)

plt.figure(figsize=(11, 9))

plt.barh(
    plot_tfidf["label"],
    plot_tfidf["mean_tfidf"],
    color="slateblue"
)

plt.xlabel("Mean TF-IDF")
plt.ylabel("Word")
plt.title(
    "Top 20 TF-IDF Terms in Official-Media Marriage Discourse"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top_20_tfidf.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. YEARLY MARRIAGE-RELATED PROPORTION
# ============================================================

# The corpus stores the publication date, not a year column.
# Derive the year so the chart below actually runs.
if (
    "year" not in df.columns
    and "date" in df.columns
):
    df["year"] = (
        df["date"]
        .fillna("")
        .astype(str)
        .str[:4]
    )

if "year" in df.columns:

    print("\n" + "=" * 70)
    print("YEARLY MARRIAGE-RELATED DOCUMENTS")
    print("=" * 70)

    yearly = (
        df.groupby("year")["marriage_related"]
        .agg(["count", "sum"])
        .reset_index()
    )

    yearly["percentage"] = (
        yearly["sum"] /
        yearly["count"] *
        100
    )

    yearly.columns = [
        "year",
        "total_documents",
        "marriage_documents",
        "percentage"
    ]

    print(
        yearly.to_string(index=False)
    )

    yearly.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "yearly_marriage_results.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Plot
    plt.figure(figsize=(9, 6))

    plt.plot(
        yearly["year"],
        yearly["percentage"],
        marker="o"
    )

    plt.xlabel("Year")
    plt.ylabel(
        "Marriage-related documents (%)"
    )

    plt.title(
        "Marriage-related Documents by Year"
    )

    plt.xticks(yearly["year"])

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "marriage_related_by_year.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 19. SAVE PROCESSED DATA
# ============================================================

# Don't save the Python lists directly.
# Convert tokens into strings first.

df_output = df.copy()

df_output["tokens"] = df_output["tokens"].apply(
    lambda x: " ".join(x)
)

df_output["tokens_clean"] = df_output[
    "tokens_clean"
].apply(
    lambda x: " ".join(x)
)


df_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "processed_corpus.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 20. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nAll results have been saved to:\n"
    f"{OUTPUT_DIR}/"
)

print("\nFiles created:")

for filename in os.listdir(OUTPUT_DIR):
    print("  -", filename)

print("\nYour first text-as-data pipeline is complete.")


# ============================================================
# 21. CHINESE FONT TEST IMAGE
# ============================================================

# Quick visual check that the chosen font really renders
# Chinese glyphs (no boxes) in a saved PNG.

print("\n==============================")
print("CHINESE FONT TEST")
print("==============================")

plt.figure(figsize=(10, 6))

plt.barh(
    ["婚姻", "家庭", "青年", "彩礼"],
    [100, 80, 60, 40],
    color="red"
)

plt.title("中文测试 Chinese Font Test")
plt.xlabel("Frequency")
plt.tight_layout()

test_path = os.path.join(
    OUTPUT_DIR,
    "CHINESE_FONT_TEST.png"
)

plt.savefig(
    test_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Chinese font test saved to: {test_path}")