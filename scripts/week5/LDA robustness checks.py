import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.colab import files
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

OUTPUT_ROOT = Path("/content/lda_robustness_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Run either or both corpora.
RUN_WEIBO = True
RUN_OFFICIAL = True

# Check 1: topic-number sensitivity.
K_VALUES = [10, 15, 20, 25]

# Check 2: random-initialisation sensitivity.
SEED_VALUES = [21, 42, 123, 2025, 2026]
REFERENCE_SEED = 42

# The teammate confirmed that the Weibo result reported as K=25 in the old script was actually based on K=15, so 15 is used here as the Weibo baseline.
WEIBO_BASELINE_K = 15

# The official-media script says K=20. Change this to 15 only if the teammate confirms that the official-media model also used K=15.
OFFICIAL_BASELINE_K = 15

TOP_N = 15

# Settings copied from the teammate's scripts. Model structure is unchanged.
CORPUS_CONFIGS = {
    "weibo": {
        "baseline_k": WEIBO_BASELINE_K,
        "min_df": 20,
        "max_df": 0.50,
        "max_features": 20000,
        "learning_method": "online",
        "batch_size": 1024,
        "max_iter": 30,
    },
    "official": {
        "baseline_k": OFFICIAL_BASELINE_K,
        "min_df": 10,
        "max_df": 0.50,
        "max_features": None,
        "learning_method": "batch",
        "batch_size": 128,  # ignored by sklearn when learning_method='batch'
        "max_iter": 50,
    },
}


def upload_one_csv(prompt):
    print("\n" + "=" * 78)
    print(prompt)
    print("=" * 78)
    uploaded = files.upload()
    csv_names = [name for name in uploaded if name.lower().endswith(".csv")]
    if len(csv_names) != 1:
        raise ValueError(
            f"Upload exactly one CSV in this dialog; found {len(csv_names)}: {csv_names}"
        )
    name = csv_names[0]
    path = Path("/content") / name
    path.write_bytes(uploaded[name])
    print(f"Received: {name}")
    return path


INPUT_FILES = {}
if RUN_WEIBO:
    INPUT_FILES["weibo"] = upload_one_csv(
        "Upload weibo_user_posts_2021_2025_preprocessed.csv"
    )
if RUN_OFFICIAL:
    INPUT_FILES["official"] = upload_one_csv(
        "Upload official_media_articles_2021_2025_preprocessed.csv"
    )



def load_corpus(path, corpus_name):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"tokens_expanded", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{corpus_name}: missing columns {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[df["tokens_expanded"].notna()].copy()
    df["tokens_expanded"] = df["tokens_expanded"].astype(str).str.strip()
    df = df[df["tokens_expanded"].str.len() > 0].reset_index(drop=True)
    df["year"] = df["year"].astype(str).str.replace(r"\.0$", "", regex=True)

    cfg = CORPUS_CONFIGS[corpus_name]
    vectorizer = CountVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        max_features=cfg["max_features"],
    )
    dtm = vectorizer.fit_transform(df["tokens_expanded"])
    vocabulary = np.asarray(vectorizer.get_feature_names_out())

    print(
        f"{corpus_name}: {dtm.shape[0]:,} documents/posts x "
        f"{dtm.shape[1]:,} vocabulary terms"
    )
    return df, dtm, vocabulary


def row_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def topic_diversity(components, top_n=TOP_N):
    top = np.argsort(components, axis=1)[:, ::-1][:, :top_n]
    return len(np.unique(top)) / float(components.shape[0] * top_n)


def npmi_coherence(dtm, components, top_n=10):
    """Document-level NPMI over each topic's top words; higher is better."""
    binary = dtm.copy().astype(bool).astype(np.int8).tocsc()
    n_docs = binary.shape[0]
    scores = []

    for component in components:
        ids = component.argsort()[::-1][:top_n]
        topic_scores = []
        for i in range(1, len(ids)):
            wi = ids[i]
            count_i = binary[:, wi].sum()
            if count_i == 0:
                continue
            for j in range(i):
                wj = ids[j]
                count_j = binary[:, wj].sum()
                both = binary[:, wi].multiply(binary[:, wj]).sum()
                if both == 0 or count_j == 0:
                    topic_scores.append(-1.0)
                    continue
                p_i = count_i / n_docs
                p_j = count_j / n_docs
                p_ij = both / n_docs
                pmi = np.log(p_ij / (p_i * p_j))
                topic_scores.append(float(pmi / (-np.log(p_ij))))
        scores.append(np.mean(topic_scores) if topic_scores else np.nan)
    return float(np.nanmean(scores)), scores


def top_word_set(component, top_n=TOP_N):
    return set(component.argsort()[::-1][:top_n].tolist())


def match_topics(reference_components, candidate_components):
    """Hungarian alignment using cosine similarity of topic-word weights."""
    ref = row_normalize(reference_components)
    cand = row_normalize(candidate_components)
    similarity = ref @ cand.T
    ref_ids, cand_ids = linear_sum_assignment(-similarity)

    rows = []
    for ref_id, cand_id in zip(ref_ids, cand_ids):
        a = top_word_set(reference_components[ref_id])
        b = top_word_set(candidate_components[cand_id])
        jaccard = len(a & b) / max(len(a | b), 1)
        rows.append({
            "reference_topic": int(ref_id + 1),
            "candidate_topic": int(cand_id + 1),
            "cosine_similarity": float(similarity[ref_id, cand_id]),
            "topword_jaccard": float(jaccard),
        })
    return rows


def yearly_topic_means(topic_weights, years):
    frame = pd.DataFrame(
        topic_weights,
        columns=[f"topic_{i}" for i in range(1, topic_weights.shape[1] + 1)],
    )
    frame["year"] = np.asarray(years)
    return frame.groupby("year").mean().sort_index()


def add_yearly_correlations(matches, ref_yearly, cand_yearly):
    common_years = ref_yearly.index.intersection(cand_yearly.index)
    for row in matches:
        ref_col = f"topic_{row['reference_topic']}"
        cand_col = f"topic_{row['candidate_topic']}"
        if len(common_years) >= 3:
            rho = spearmanr(
                ref_yearly.loc[common_years, ref_col],
                cand_yearly.loc[common_years, cand_col],
            ).statistic
            row["yearly_prevalence_spearman"] = float(rho) if np.isfinite(rho) else np.nan
        else:
            row["yearly_prevalence_spearman"] = np.nan
    return matches


def make_model(corpus_name, k, seed):
    cfg = CORPUS_CONFIGS[corpus_name]
    kwargs = dict(
        n_components=k,
        random_state=seed,
        learning_method=cfg["learning_method"],
        max_iter=cfg["max_iter"],
        evaluate_every=-1,
        n_jobs=-1,
    )
    if cfg["learning_method"] == "online":
        kwargs["batch_size"] = cfg["batch_size"]
    return LatentDirichletAllocation(**kwargs)


def fit_run(corpus_name, df, dtm, vocabulary, k, seed, out_dir):
    print(f"[{corpus_name}] fitting K={k}, seed={seed} ...", flush=True)
    start = time.time()
    model = make_model(corpus_name, k, seed)
    weights = model.fit_transform(dtm)
    elapsed = time.time() - start

    coherence_mean, coherence_by_topic = npmi_coherence(dtm, model.components_)
    diversity = topic_diversity(model.components_)
    perplexity = float(model.perplexity(dtm))  # descriptive in-sample metric

    run_id = f"k{k}_seed{seed}"
    topic_rows = []
    for i, component in enumerate(model.components_, start=1):
        ids = component.argsort()[::-1][:TOP_N]
        topic_rows.append({
            "corpus": corpus_name,
            "run_id": run_id,
            "k": k,
            "seed": seed,
            "topic": i,
            "top_words": " ".join(vocabulary[ids]),
            "npmi_coherence": coherence_by_topic[i - 1],
        })

    pd.DataFrame(topic_rows).to_csv(
        out_dir / f"topics_{run_id}.csv", index=False, encoding="utf-8-sig"
    )
    yearly = yearly_topic_means(weights, df["year"].values)
    yearly.to_csv(out_dir / f"topic_by_year_{run_id}.csv", encoding="utf-8-sig")

    metric = {
        "corpus": corpus_name,
        "run_id": run_id,
        "k": k,
        "seed": seed,
        "n_documents": dtm.shape[0],
        "n_vocabulary": dtm.shape[1],
        "learning_method": CORPUS_CONFIGS[corpus_name]["learning_method"],
        "max_iter": CORPUS_CONFIGS[corpus_name]["max_iter"],
        "in_sample_perplexity": perplexity,
        "mean_npmi_coherence": coherence_mean,
        "topic_diversity": diversity,
        "elapsed_seconds": elapsed,
    }
    print(
        f"  done: NPMI={coherence_mean:.3f}, diversity={diversity:.3f}, "
        f"perplexity={perplexity:.1f}, time={elapsed/60:.1f} min"
    )
    return {
        "model": model,
        "weights": weights,
        "yearly": yearly,
        "metric": metric,
    }


def run_corpus(corpus_name, input_path):
    cfg = CORPUS_CONFIGS[corpus_name]
    baseline_k = cfg["baseline_k"]
    out_dir = OUTPUT_ROOT / corpus_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df, dtm, vocabulary = load_corpus(input_path, corpus_name)

    # Avoid fitting the shared baseline run twice.
    run_pairs = {(k, REFERENCE_SEED) for k in K_VALUES}
    run_pairs |= {(baseline_k, seed) for seed in SEED_VALUES}
    run_pairs = sorted(run_pairs)

    fitted = {}
    for k, seed in run_pairs:
        fitted[(k, seed)] = fit_run(
            corpus_name, df, dtm, vocabulary, k, seed, out_dir
        )

    metrics = pd.DataFrame([fitted[key]["metric"] for key in run_pairs])
    metrics.to_csv(out_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")

    reference = fitted[(baseline_k, REFERENCE_SEED)]

    # Check 1: match each alternative K to the baseline-K topics.
    k_matches = []
    for k in K_VALUES:
        candidate = fitted[(k, REFERENCE_SEED)]
        matches = match_topics(
            reference["model"].components_, candidate["model"].components_
        )
        matches = add_yearly_correlations(
            matches, reference["yearly"], candidate["yearly"]
        )
        for row in matches:
            row.update({
                "corpus": corpus_name,
                "check": "alternative_k",
                "reference_k": baseline_k,
                "candidate_k": k,
                "reference_seed": REFERENCE_SEED,
                "candidate_seed": REFERENCE_SEED,
            })
            k_matches.append(row)
    pd.DataFrame(k_matches).to_csv(
        out_dir / "check1_k_topic_matches.csv", index=False, encoding="utf-8-sig"
    )

    # Check 2: match alternative seeds to seed 42 at the baseline K.
    seed_matches = []
    for seed in SEED_VALUES:
        candidate = fitted[(baseline_k, seed)]
        matches = match_topics(
            reference["model"].components_, candidate["model"].components_
        )
        matches = add_yearly_correlations(
            matches, reference["yearly"], candidate["yearly"]
        )
        for row in matches:
            row.update({
                "corpus": corpus_name,
                "check": "alternative_seed",
                "reference_k": baseline_k,
                "candidate_k": baseline_k,
                "reference_seed": REFERENCE_SEED,
                "candidate_seed": seed,
            })
            seed_matches.append(row)
    seed_matches_df = pd.DataFrame(seed_matches)
    seed_matches_df.to_csv(
        out_dir / "check2_seed_topic_matches.csv", index=False, encoding="utf-8-sig"
    )

    seed_summary = (
        seed_matches_df.groupby("candidate_seed", as_index=False)
        .agg(
            mean_topic_cosine=("cosine_similarity", "mean"),
            minimum_topic_cosine=("cosine_similarity", "min"),
            mean_topword_jaccard=("topword_jaccard", "mean"),
            mean_yearly_spearman=("yearly_prevalence_spearman", "mean"),
        )
    )
    seed_summary.to_csv(
        out_dir / "check2_seed_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out_dir / "run_settings.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "corpus": corpus_name,
                "input_file": input_path.name,
                "k_values": K_VALUES,
                "baseline_k": baseline_k,
                "seed_values": SEED_VALUES,
                "reference_seed": REFERENCE_SEED,
                **cfg,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[{corpus_name}] results saved to {out_dir}")


for corpus_name, input_path in INPUT_FILES.items():
    run_corpus(corpus_name, input_path)



