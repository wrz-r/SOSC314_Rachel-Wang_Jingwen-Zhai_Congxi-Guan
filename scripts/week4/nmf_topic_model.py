# -*- coding: utf-8 -*-
"""TF-IDF and NMF topic modeling for official-media marriage articles."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "official_media_articles_2021_2025_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "nmf_final_results"
OUTPUT_DIR.mkdir(exist_ok=True)

TOKEN_COLUMN = "tokens_expanded"
K_GRID = list(range(4, 31, 2))

MIN_DF = 10
MAX_DF = 0.50
MAX_FEATURES = None

TOP_WORDS = 15
COHERENCE_TOP_N = 10
RANDOM_SEED = 42

plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "STHeiti"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# STEP 1. LOAD PREPROCESSED ARTICLES
# ============================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

required = {"media", "date", "title", "url", TOKEN_COLUMN}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

df = df[df[TOKEN_COLUMN].notna()].copy()
df[TOKEN_COLUMN] = df[TOKEN_COLUMN].astype(str).str.strip()
df = df[df[TOKEN_COLUMN].str.len() > 0].reset_index(drop=True)

if "year" not in df.columns:
    df["year"] = df["date"].astype(str).str[:4]
else:
    df["year"] = df["year"].astype(str).str.replace(r"\.0$", "", regex=True)

print(f"Documents: {len(df):,}")
print(f"Years: {sorted(df['year'].unique())}")


# ============================================================
# STEP 2. BUILD TF-IDF MATRIX
# NMF commonly uses TF-IDF rather than raw word counts.
# ============================================================

vectorizer = TfidfVectorizer(
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    min_df=MIN_DF,
    max_df=MAX_DF,
    max_features=MAX_FEATURES,
    sublinear_tf=True,
)

tfidf = vectorizer.fit_transform(df[TOKEN_COLUMN])
vocabulary = np.asarray(vectorizer.get_feature_names_out())
print(f"TF-IDF matrix: {tfidf.shape[0]:,} documents x {tfidf.shape[1]:,} words")

# A binary word-presence matrix is used only for UMass coherence.
count_vectorizer = CountVectorizer(
    vocabulary=vectorizer.vocabulary_,
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    binary=True,
)
binary_dtm = count_vectorizer.transform(df[TOKEN_COLUMN]).astype(np.int32)
word_document_frequency = np.asarray(binary_dtm.sum(axis=0)).ravel()


def umass_coherence(topic_weights):
    """Calculate UMass coherence; higher values are better."""
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
# STEP 3. COMPARE TOPIC NUMBERS
# ============================================================

model_results = []

for k in K_GRID:
    model = NMF(
        n_components=k,
        init="nndsvda",
        random_state=RANDOM_SEED,
        max_iter=500,
    )
    model.fit(tfidf)

    coherence = np.mean(
        [umass_coherence(topic) for topic in model.components_]
    )
    model_results.append(
        {
            "k": k,
            "coherence": coherence,
            "reconstruction_error": model.reconstruction_err_,
        }
    )
    print(
        f"K={k:2d} | coherence={coherence:.4f} | "
        f"reconstruction error={model.reconstruction_err_:.4f}",
        flush=True,
    )

comparison = pd.DataFrame(model_results)
comparison.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

# Use coherence to select K. Reconstruction error is also saved for comparison.
best_k = int(comparison.loc[comparison["coherence"].idxmax(), "k"])
print(f"\nSelected K = {best_k}")


# ============================================================
# STEP 4. RUN FINAL NMF MODEL
# ============================================================

final_model = NMF(
    n_components=best_k,
    init="nndsvda",
    random_state=RANDOM_SEED,
    max_iter=1000,
)
document_topics = final_model.fit_transform(tfidf)

# Convert topic weights into proportions that sum to 1 for each article.
row_sums = document_topics.sum(axis=1, keepdims=True)
document_topics = np.divide(
    document_topics,
    row_sums,
    out=np.zeros_like(document_topics),
    where=row_sums != 0,
)


# ============================================================
# STEP 5. SAVE TOP WORDS
# ============================================================

topic_rows = []
for topic_number, weights in enumerate(final_model.components_, start=1):
    top_indices = weights.argsort()[::-1][:TOP_WORDS]
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
# STEP 6. SAVE ARTICLE TOPICS AND YEARLY TOPIC SHARES
# ============================================================

topic_columns = [f"topic_{i}" for i in range(1, best_k + 1)]
distribution = pd.DataFrame(document_topics, columns=topic_columns)
distribution.insert(0, "dominant_topic", document_topics.argmax(axis=1) + 1)

metadata_columns = [
    column
    for column in ["media", "date", "year", "title", "url", "matched_keywords"]
    if column in df.columns
]
document_output = pd.concat(
    [df[metadata_columns].reset_index(drop=True), distribution],
    axis=1,
)
document_output.to_csv(
    OUTPUT_DIR / "document_topic_distribution.csv",
    index=False,
    encoding="utf-8-sig",
)

yearly = distribution[topic_columns].copy()
yearly["year"] = df["year"].values
topic_by_year = yearly.groupby("year").mean().sort_index()
topic_by_year.to_csv(
    OUTPUT_DIR / "topic_by_year.csv",
    encoding="utf-8-sig",
)

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
ax.set_title("NMF official-media marriage topics, 2021-2025")
ax.legend(topic_labels, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "topic_by_year.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nAll results saved to: {OUTPUT_DIR}")
