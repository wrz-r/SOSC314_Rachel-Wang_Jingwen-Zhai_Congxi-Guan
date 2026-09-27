# -*- coding: utf-8 -*-
"""Bag-of-words and LDA topic modeling for official-media articles."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "official_media_articles_2021_2025_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "lda_qualitative_k_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TOKEN_COLUMN = "tokens_expanded"

# Compare several possible topic numbers.
K_VALUES = [10, 15, 20, 25, 30]
FINAL_K = 20

# Bag-of-words vocabulary rules.
MIN_DF = 10       # A word must appear in at least 10 articles.
MAX_DF = 0.50     # Remove words appearing in more than 50% of articles.
MAX_FEATURES = None  # None means no fixed maximum vocabulary size.

TOP_WORDS = 15
REPRESENTATIVE_ARTICLES = 3
ARTICLE_PREVIEW_LENGTH = 250
RANDOM_SEED = 42


# ============================================================
# STEP 1. LOAD THE PREPROCESSED ARTICLES
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = {"media", "date", "title", "url", TOKEN_COLUMN}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

df = df[df[TOKEN_COLUMN].notna()].copy()
df[TOKEN_COLUMN] = df[TOKEN_COLUMN].astype(str).str.strip()
df = df[df[TOKEN_COLUMN].str.len() > 0].reset_index(drop=True)

# Create the year variable if it is absent.
if "year" not in df.columns:
    df["year"] = df["date"].astype(str).str[:4]
else:
    df["year"] = df["year"].astype(str).str.replace(r"\.0$", "", regex=True)

print(f"Documents: {len(df):,}")
print(f"Years: {sorted(df['year'].unique())}")


# ============================================================
# STEP 2. BUILD THE BAG-OF-WORDS MATRIX
# ============================================================

vectorizer = CountVectorizer(
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
)

dtm = vectorizer.fit_transform(df[TOKEN_COLUMN])
vocabulary = np.asarray(vectorizer.get_feature_names_out())

print(f"BOW matrix: {dtm.shape[0]:,} documents x {dtm.shape[1]:,} words")

# ============================================================
# STEP 3. COMPARE DIFFERENT NUMBERS OF TOPICS
# ============================================================

comparison_topic_rows = []

for k in K_VALUES:
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=RANDOM_SEED,
        learning_method="batch",
        max_iter=20,
        n_jobs=-1,
    )
    document_topic_weights = lda.fit_transform(dtm)

    # Save topic words and the highest-weighted articles for qualitative review.
    for topic_number, topic_weights in enumerate(lda.components_, start=1):
        top_indices = topic_weights.argsort()[::-1][:TOP_WORDS]
        top_document_indices = np.argsort(
            document_topic_weights[:, topic_number - 1]
        )[::-1][:REPRESENTATIVE_ARTICLES]

        row = {
            "k": k,
            "topic": topic_number,
            "top_words": " ".join(vocabulary[top_indices]),
        }

        for rank, document_index in enumerate(top_document_indices, start=1):
            article = df.iloc[document_index]
            article_text = str(article.get("text", ""))
            row[f"representative_article_{rank}"] = article["title"]
            row[f"representative_excerpt_{rank}"] = article_text[:ARTICLE_PREVIEW_LENGTH]

        # Fill in these columns after reading the topic words and articles.
        row["topic_interpretation"] = ""
        row["clear_and_distinct"] = ""
        row["notes"] = ""
        comparison_topic_rows.append(row)

    print(f"K={k:2d} completed", flush=True)

comparison_topics = pd.DataFrame(comparison_topic_rows)
comparison_topics.to_csv(
    OUTPUT_DIR / "topics_for_k_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

if FINAL_K not in K_VALUES:
    raise ValueError("FINAL_K must be one of the values in K_VALUES")

print(f"\nManually selected FINAL_K = {FINAL_K}")
print("Review topics_for_k_comparison.csv and then change FINAL_K if needed.")
print("Choose the smallest K with clear, distinct and substantively useful topics.")

# ============================================================
# STEP 4. RUN THE MANUALLY SELECTED FINAL LDA MODEL
# ============================================================

final_lda = LatentDirichletAllocation(
    n_components=FINAL_K,
    random_state=RANDOM_SEED,
    learning_method="batch",
    max_iter=50,
    n_jobs=-1,
)

document_topics = final_lda.fit_transform(dtm)
print("Final LDA model completed.")


# ============================================================
# STEP 5. SAVE THE TOP WORDS FOR THE SELECTED MODEL
# ============================================================

topic_rows = []

for topic_number, topic_weights in enumerate(final_lda.components_, start=1):
    top_indices = topic_weights.argsort()[::-1][:TOP_WORDS]
    top_words = vocabulary[top_indices]

    topic_rows.append(
        {
            "topic": topic_number,
            "top_words": " ".join(top_words),
        }
    )

topics_df = pd.DataFrame(topic_rows)
topics_df.to_csv(
    OUTPUT_DIR / "topics_top_words.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nTop words")
for _, row in topics_df.iterrows():
    print(f"Topic {row['topic']}: {row['top_words']}")


# ============================================================
# STEP 6. SAVE EACH ARTICLE'S TOPIC DISTRIBUTION
# ============================================================

topic_columns = [f"topic_{i}" for i in range(1, FINAL_K + 1)]
distribution_df = pd.DataFrame(document_topics, columns=topic_columns)
distribution_df.insert(0, "dominant_topic", document_topics.argmax(axis=1) + 1)

metadata_columns = [
    column
    for column in ["media", "date", "year", "title", "url", "matched_keywords"]
    if column in df.columns
]

document_output = pd.concat(
    [df[metadata_columns].reset_index(drop=True), distribution_df],
    axis=1,
)
document_output.to_csv(
    OUTPUT_DIR / "document_topic_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# STEP 7. COMPARE TOPIC PREVALENCE BY YEAR
# ============================================================

yearly_topics = distribution_df[topic_columns].copy()
yearly_topics["year"] = df["year"].values
topic_by_year = yearly_topics.groupby("year").mean().sort_index()

topic_by_year.to_csv(
    OUTPUT_DIR / "topic_by_year.csv",
    encoding="utf-8-sig",
)


print(f"\nAll results saved to: {OUTPUT_DIR}")
