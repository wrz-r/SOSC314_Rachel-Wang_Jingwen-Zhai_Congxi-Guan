# -*- coding: utf-8 -*-
"""Bag-of-words and LDA topic modeling for official-media articles."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "official_media_articles_2021_2025_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "lda_final_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TOKEN_COLUMN = "tokens_expanded"

# Compare K = 4, 6, 8, ..., 30.
K_GRID = list(range(4, 31, 2))

# Bag-of-words vocabulary rules.
MIN_DF = 10       # A word must appear in at least 10 articles.
MAX_DF = 0.50     # Remove words appearing in more than 50% of articles.
MAX_FEATURES = None  # None means no fixed maximum vocabulary size.

TOP_WORDS = 15
COHERENCE_TOP_N = 10
RANDOM_SEED = 42


# Chinese fonts for figures.
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# STEP 1. LOAD THE PREPROCESSED ARTICLES
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required_columns = {"media", "date", "title", "url", TOKEN_COLUMN}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {', '.join(sorted(missing_columns))}")

# Keep articles that have usable expanded tokens.
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

# tokens_expanded is already segmented by jieba and separated by spaces.
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
# STEP 3. CALCULATE UMass COHERENCE
# Higher values, closer to zero, indicate better coherence.
# ============================================================

binary_dtm = (dtm > 0).astype(np.int32)
word_document_frequency = np.asarray(binary_dtm.sum(axis=0)).ravel()


def umass_coherence(topic_weights):
    """Calculate UMass coherence for one topic."""
    top_indices = topic_weights.argsort()[::-1][:COHERENCE_TOP_N]
    scores = []

    for i in range(1, len(top_indices)):
        word_i = top_indices[i]
        for j in range(i):
            word_j = top_indices[j]
            co_documents = int(
                binary_dtm[:, word_i]
                .multiply(binary_dtm[:, word_j])
                .sum()
            )
            scores.append(
                np.log(
                    (co_documents + 1.0)
                    / max(word_document_frequency[word_j], 1)
                )
            )

    return float(np.mean(scores))


# ============================================================
# STEP 4. COMPARE DIFFERENT NUMBERS OF TOPICS
# ============================================================

model_results = []

for k in K_GRID:
    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=RANDOM_SEED,
        learning_method="batch",
        max_iter=20,
        n_jobs=-1,
    )
    lda.fit(dtm)

    coherence = np.mean(
        [umass_coherence(topic) for topic in lda.components_]
    )
    perplexity = lda.perplexity(dtm)

    model_results.append(
        {"k": k, "coherence": coherence, "perplexity": perplexity}
    )
    print(
        f"K={k:2d} | coherence={coherence:.4f} | "
        f"perplexity={perplexity:.2f}",
        flush=True,
    )

comparison = pd.DataFrame(model_results)
comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

# Select the K with the highest coherence score.
best_k = int(comparison.loc[comparison["coherence"].idxmax(), "k"])
print(f"\nSelected K = {best_k}")

# ============================================================
# STEP 5. RUN THE FINAL LDA MODEL
# ============================================================

final_lda = LatentDirichletAllocation(
    n_components=best_k,
    random_state=RANDOM_SEED,
    learning_method="batch",
    max_iter=50,
    n_jobs=-1,
)

document_topics = final_lda.fit_transform(dtm)
print(f"Final model perplexity: {final_lda.perplexity(dtm):.2f}")


# ============================================================
# STEP 6. SAVE THE TOP WORDS FOR EACH TOPIC
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
# STEP 7. SAVE EACH ARTICLE'S TOPIC DISTRIBUTION
# ============================================================

topic_columns = [f"topic_{i}" for i in range(1, best_k + 1)]
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
# STEP 8. COMPARE TOPIC PREVALENCE BY YEAR
# ============================================================

yearly_topics = distribution_df[topic_columns].copy()
yearly_topics["year"] = df["year"].values
topic_by_year = yearly_topics.groupby("year").mean().sort_index()

topic_by_year.to_csv(
    OUTPUT_DIR / "topic_by_year.csv",
    encoding="utf-8-sig",
)

# Use each topic's first three words as its chart label.
topic_labels = [
    f"T{int(row['topic'])} " + " · ".join(row["top_words"].split()[:3])
    for _, row in topics_df.iterrows()
]

ax = topic_by_year.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7),
    colormap="tab20",
)
ax.set_xlabel("Year")
ax.set_ylabel("Mean topic share")
ax.set_title("Official-media marriage topics, 2021-2025")
ax.legend(topic_labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "topic_by_year.png", dpi=200, bbox_inches="tight")
plt.close()


print(f"\nAll results saved to: {OUTPUT_DIR}")
