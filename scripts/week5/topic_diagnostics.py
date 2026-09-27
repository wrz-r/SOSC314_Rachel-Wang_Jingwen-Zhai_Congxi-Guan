# -*- coding: utf-8 -*-
"""Check coherence, exclusivity, and representative texts for both LDA models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


BASE = Path(__file__).resolve().parent
K = 15
TOP_WORDS = 15
N_WORDS = 10          # Use the same number of words for both diagnostics.
N_EXAMPLES = 5        # Read five high-probability texts per topic.

CORPORA = {
    "official": {
        "input": BASE / "official_media_articles_2021_2025_preprocessed.csv",
        "results": BASE / "lda_qualitative_k_results",
        "min_df": 10,
        "max_features": None,
        "lda": {"learning_method": "batch", "max_iter": 50},
        "text_column": "text",
        "details": ["media", "date", "title", "url"],
    },
    "weibo": {
        "input": BASE / "weibo_user_posts_2021_2025_preprocessed.csv",
        "results": BASE / "weibo_lda_results",
        "min_df": 20,
        "max_features": 20000,
        "lda": {
            "learning_method": "online", "batch_size": 1024,
            "max_iter": 30, "evaluate_every": -1,
        },
        "text_column": "clean_text",
        "details": ["发布时间", "id", "query_keyword"],
    },
}


def analyze(name):
    settings = CORPORA[name]
    print(f"\n{name}: reading data", flush=True)
    data = pd.read_csv(
        settings["input"], encoding="utf-8-sig", low_memory=False,
        dtype={"id": "string"} if name == "weibo" else None,
    )
    data = data[data["tokens_expanded"].notna()].copy()
    data["tokens_expanded"] = data["tokens_expanded"].astype(str).str.strip()
    data = data[data["tokens_expanded"].str.len() > 0].reset_index(drop=True)

    vectorizer = CountVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        min_df=settings["min_df"], max_df=0.50,
        max_features=settings["max_features"],
    )
    matrix = vectorizer.fit_transform(data["tokens_expanded"])
    vocabulary = vectorizer.get_feature_names_out()

    # The earlier scripts saved topic words, but not the full topic-word weights.
    # Fit once with their final-model settings to recover those weights.
    model = LatentDirichletAllocation(
        n_components=K, random_state=42, n_jobs=-1, **settings["lda"]
    )
    print(f"{name}: fitting K={K} model", flush=True)
    document_topics = model.fit_transform(matrix)
    word_weights = model.components_ / model.components_.sum(axis=1, keepdims=True)

    saved = pd.read_csv(settings["results"] / "topics_top_words.csv")
    if len(saved) != K:
        raise ValueError(f"{name}: saved topics are not K={K}.")

    top_ids_by_topic = []
    topic_words = []
    for topic_index in range(K):
        top_ids = np.argsort(model.components_[topic_index])[::-1][:TOP_WORDS]
        words = list(vocabulary[top_ids])
        saved_words = saved.loc[saved["topic"] == topic_index + 1, "top_words"]
        if len(saved_words) != 1 or " ".join(words) != str(saved_words.iloc[0]):
            raise ValueError(f"{name}: topic {topic_index + 1} differs from the saved model."
                             " Check the input and LDA settings before using these scores.")
        top_ids_by_topic.append(top_ids)
        topic_words.append(words)

    # Gensim c_v coherence uses word co-occurrence in sliding text windows.
    texts = data["tokens_expanded"].str.split().tolist()
    coherence = CoherenceModel(
        topics=[words[:N_WORDS] for words in topic_words],
        texts=texts, dictionary=Dictionary(texts),
        coherence="c_v", topn=N_WORDS, processes=1,
    ).get_coherence_per_topic()

    summary_rows = []
    review_rows = []

    for topic_index in range(K):
        topic = topic_index + 1
        selected = top_ids_by_topic[topic_index][:N_WORDS]

        # For each top word: share of its weight belonging to this topic.
        # This simple 0–1 measure follows Topica's exclusivity idea;
        # it is not Topica's FREX-based exclusivity statistic.
        word_shares = word_weights[topic_index, selected] / word_weights[:, selected].sum(axis=0)
        exclusivity = float(word_shares.mean())

        summary_rows.append({
            "topic": topic,
            "top_words": " ".join(topic_words[topic_index]),
            "coherence_c_v": round(float(coherence[topic_index]), 3),
            "exclusivity_mean": round(exclusivity, 3),
            "topic_label": "",  # Fill after reading the sample texts.
        })

        # A document may appear for more than one topic: LDA allows topic mixtures.
        best_documents = np.argsort(document_topics[:, topic_index])[::-1][:N_EXAMPLES]
        for rank, row_index in enumerate(best_documents, start=1):
            source = data.iloc[row_index]
            row = {
                "topic": topic,
                "rank": rank,
                "topic_probability": round(float(document_topics[row_index, topic_index]), 3),
                "text": source.get(settings["text_column"], ""),
            }
            row.update({col: source.get(col, "") for col in settings["details"]})
            row.update({"fits_topic": "", "notes": ""})  # Fill by hand: yes/no + reason.
            review_rows.append(row)

    result_dir = settings["results"]
    pd.DataFrame(summary_rows).to_csv(
        result_dir / "topic_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(review_rows).to_csv(
        result_dir / "topic_review_samples.csv", index=False, encoding="utf-8-sig"
    )
    print(f"{name}: saved topic_diagnostics.csv and topic_review_samples.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDA topic diagnostics")
    parser.add_argument("--corpus", choices=["official", "weibo", "both"], default="both")
    choice = parser.parse_args().corpus
    for corpus in CORPORA if choice == "both" else [choice]:
        analyze(corpus)
