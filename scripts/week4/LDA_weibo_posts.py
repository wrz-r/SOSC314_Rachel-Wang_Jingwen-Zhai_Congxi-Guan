# -*- coding: utf-8 -*-
"""Bag-of-words and LDA topic modeling for user-initiated Weibo posts."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "weibo_user_posts_2021_2025_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "weibo_lda_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TOKEN_COLUMN = "tokens_expanded"
TEXT_COLUMN = "clean_text"

K_VALUES = [10, 15, 20, 25, 30]
FINAL_K = 25

# Bag-of-words settings.
MIN_DF = 20
MAX_DF = 0.50
MAX_FEATURES = 20000

TOP_WORDS = 15
REPRESENTATIVE_POSTS = 3
POST_PREVIEW_LENGTH = 250
RANDOM_SEED = 42


# ============================================================
# STEP 1. LOAD THE PREPROCESSED POSTS
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig", low_memory=False)

required_columns = {TOKEN_COLUMN, TEXT_COLUMN, "year"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

df = df[df[TOKEN_COLUMN].notna()].copy()
df[TOKEN_COLUMN] = df[TOKEN_COLUMN].astype(str).str.strip()
df = df[df[TOKEN_COLUMN].str.len() > 0].reset_index(drop=True)
df["year"] = df["year"].astype(str).str.replace(r"\.0$", "", regex=True)

print(f"Posts: {len(df):,}")
print(f"Years: {sorted(df['year'].dropna().unique())}")


# ============================================================
# STEP 2. BUILD THE BAG-OF-WORDS MATRIX
# ============================================================

# The posts have already been segmented; tokens are separated by spaces.
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

print(f"BOW matrix: {dtm.shape[0]:,} posts x {dtm.shape[1]:,} words")


def make_lda(k: int, max_iter: int) -> LatentDirichletAllocation:
    """Create an LDA model suitable for the larger Weibo corpus."""
    return LatentDirichletAllocation(
        n_components=k,
        random_state=RANDOM_SEED,
        learning_method="online",
        batch_size=1024,
        max_iter=max_iter,
        evaluate_every=-1,
        n_jobs=-1,
    )


# ============================================================
# STEP 3. COMPARE DIFFERENT NUMBERS OF TOPICS
# ============================================================

comparison_rows = []

for k in K_VALUES:
    model = make_lda(k, max_iter=10)
    post_topic_weights = model.fit_transform(dtm)

    for topic_number, topic_weights in enumerate(model.components_, start=1):
        top_word_indices = topic_weights.argsort()[::-1][:TOP_WORDS]
        top_post_indices = np.argsort(
            post_topic_weights[:, topic_number - 1]
        )[::-1][:REPRESENTATIVE_POSTS]

        row = {
            "k": k,
            "topic": topic_number,
            "top_words": " ".join(vocabulary[top_word_indices]),
        }

        for rank, post_index in enumerate(top_post_indices, start=1):
            post_text = str(df.iloc[post_index][TEXT_COLUMN])
            row[f"representative_post_{rank}"] = post_text[:POST_PREVIEW_LENGTH]

        # Complete these columns when manually comparing the models.
        row["topic_interpretation"] = ""
        row["clear_and_distinct"] = ""
        row["notes"] = ""
        comparison_rows.append(row)

    print(f"K={k:2d} completed", flush=True)

pd.DataFrame(comparison_rows).to_csv(
    OUTPUT_DIR / "topics_for_k_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

if FINAL_K not in K_VALUES:
    raise ValueError("FINAL_K must be one of the values in K_VALUES")

print(f"\nProvisional FINAL_K = {FINAL_K}")
print("Review topics_for_k_comparison.csv before confirming the final K.")


# ============================================================
# STEP 4. RUN THE FINAL LDA MODEL
# ============================================================

final_model = make_lda(FINAL_K, max_iter=30)
post_topics = final_model.fit_transform(dtm)
print("Final LDA model completed.")


# ============================================================
# STEP 5. SAVE THE FINAL TOPICS
# ============================================================

topic_rows = []

for topic_number, topic_weights in enumerate(final_model.components_, start=1):
    top_indices = topic_weights.argsort()[::-1][:TOP_WORDS]
    topic_rows.append(
        {
            "topic": topic_number,
            "top_words": " ".join(vocabulary[top_indices]),
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
# STEP 6. SAVE EACH POST'S TOPIC DISTRIBUTION
# ============================================================

topic_columns = [f"topic_{i}" for i in range(1, FINAL_K + 1)]
distribution_df = pd.DataFrame(post_topics, columns=topic_columns)
distribution_df.insert(0, "dominant_topic", post_topics.argmax(axis=1) + 1)

metadata_columns = [
    column
    for column in [
        "sampling_year", "发布时间", "id", "user_id", "query_keyword",
        "微博正文", "relevance_level", "relevance_rule",
    ]
    if column in df.columns
]

post_output = pd.concat(
    [df[metadata_columns].reset_index(drop=True), distribution_df],
    axis=1,
)
post_output.to_csv(
    OUTPUT_DIR / "post_topic_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# STEP 7. SAVE TOPIC PROPORTIONS BY YEAR
# ============================================================

yearly_topics = distribution_df[topic_columns].copy()
yearly_topics["year"] = df["year"].values
topic_by_year = yearly_topics.groupby("year").mean().sort_index()

topic_by_year.to_csv(
    OUTPUT_DIR / "topic_by_year.csv",
    encoding="utf-8-sig",
)


print(f"\nResults saved to: {OUTPUT_DIR}")
print("1. topics_for_k_comparison.csv")
print("2. topics_top_words.csv")
print("3. post_topic_distribution.csv")
print("4. topic_by_year.csv")
