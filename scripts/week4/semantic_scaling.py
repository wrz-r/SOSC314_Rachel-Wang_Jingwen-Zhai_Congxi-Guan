"""
============================================================================
 Semantic Scaling for attitude measurement
============================================================================

"""

# =============================================================================
# Part 0 — Imports & config
# =============================================================================
import os
import csv
import re
import json
from collections import defaultdict, Counter

import numpy as np
import jieba
from sentence_transformers import SentenceTransformer

CORPUS  = "/Users/congxi/model/official_media_articles_2021_2025_merged.csv"
OUT_DIR = "/Users/congxi/model/attitude_analysis/output"

EMBED_MODEL = "shibing624/text2vec-base-chinese"   # 768-d, Chinese, ~400MB
BATCH_SIZE  = 64                                    # embedding batch size

# --- Seed anchors (Chinese) ---
POS_SEEDS = ["幸福", "美满", "和谐", "陪伴", "稳定"]
NEG_SEEDS = ["压力", "负担", "痛苦", "束缚", "恐惧"]

# Stopwords / tokens we don't want contributing to doc scores.
STOPWORDS = set("""
的 了 在 是 我 你 他 她 它 我们 你们 他们 和 与 及 或 也 都 就 还
这 那 此 其 之 于 以 对 把 被 让 使 给 向 从 到 为 而 但 然而
一个 一些 这 那 一 不 没 没有 很 太 非常 比较 更 最 都 也 还 已经
将 把 将 word 并 并且 或者 而且 因为 所以 如果 虽然 但是 然后 因为
""".split())


# =============================================================================
# Part 1 — Tokenisation
# =============================================================================
def tokenize(text):
    """Chinese word segmentation via jieba; drop stopwords & pure punctuation."""
    if not text:
        return []
    # jieba.cut returns an iterator of tokens
    toks = [t.strip() for t in jieba.cut(text)]
    return [t for t in toks
            if t
            and t not in STOPWORDS
            and not re.fullmatch(r"[\W_]+", t)
            and len(t) >= 2]


# =============================================================================
# Part 2 — Embed seeds and corpus words
# =============================================================================
def embed_texts(model, texts, batch_size=BATCH_SIZE):
    """Run SentenceTransformer in batches; returns L2-normalised vectors."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise so dot = cosine
        convert_to_numpy=True,
    )
    return vecs


# =============================================================================
# Part 3 — Build the semantic axis & score words
# =============================================================================
def build_axis(pos_emb, neg_emb):
    """
    The semantic "attitude axis" is the vector from negative pole to
    positive pole in embedding space.
    """
    pos_centroid = pos_emb.mean(axis=0)
    neg_centroid = neg_emb.mean(axis=0)
    direction = pos_centroid - neg_centroid
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction


def project(words_emb, direction):
    """
    Scalar projection of each word onto the axis.
    word_score = dot(word_emb, direction)
    Because embeddings are L2-normalised and direction is unit-norm,
    this is equivalent to cosine similarity with the axis.
    """
    return words_emb @ direction


def z_calibrate(scores, seed_mask, target_std=1.0):
    """
    Re-scale raw projection scores so that the seed words have unit
    spread. This puts scores on a roughly [-1, +1] scale.
    """
    seed_scores = scores[seed_mask]
    mu  = seed_scores.mean()
    sig = seed_scores.std() + 1e-12
    return (scores - mu) / sig * target_std


# =============================================================================
# Part 4 — Score each document as the mean of its word scores
# =============================================================================
def doc_score(tokens, word2score):
    """Mean of in-vocab word scores for this document; missing words skipped."""
    vals = [word2score[t] for t in tokens if t in word2score]
    if not vals:
        return 0.0
    return float(np.mean(vals))


# =============================================================================
# Part 5 — Main
# =============================================================================
def main():
    print(f"\n[embed] loading {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    # ----- 5.1 Embed seed anchors -----
    print("\n[seeds] embedding seed anchors ...")
    pos_emb = embed_texts(model, POS_SEEDS, batch_size=len(POS_SEEDS))
    neg_emb = embed_texts(model, NEG_SEEDS, batch_size=len(NEG_SEEDS))

    # ----- 5.2 Build the semantic axis -----
    direction = build_axis(pos_emb, neg_emb)
    seed_all = np.concatenate([pos_emb, neg_emb])
    seed_mask_idx = list(range(len(POS_SEEDS) + len(NEG_SEEDS)))

    # ----- 5.3 Tokenise corpus & collect unique words -----
    print("\n[corpus] tokenising ...")
    with open(CORPUS, encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))

    docs_tokens = []
    for r in corpus:
        text = r.get("text", "") or ""
        docs_tokens.append(tokenize(text))

    # Count vocabulary across the corpus
    vocab_counter = Counter()
    for toks in docs_tokens:
        vocab_counter.update(toks)
    # Keep words that appear in >= MIN_DF documents (computed approx via total freq).
    MIN_FREQ = 3
    vocab = [w for w, c in vocab_counter.items() if c >= MIN_FREQ]
    print(f"[vocab] {len(vocab_counter)} unique tokens, {len(vocab)} kept (freq>={MIN_FREQ})")

    # ----- 5.4 Embed vocabulary words -----
    print("\n[embed] embedding vocabulary ...")
    vocab_emb = embed_texts(model, vocab, batch_size=BATCH_SIZE)

    # ----- 5.5 Project every word onto the axis -----
    raw_scores = project(vocab_emb, direction)
    seed_in_vocab_mask = np.array([w in (POS_SEEDS + NEG_SEEDS) for w in vocab])
    calibrated = z_calibrate(raw_scores, seed_in_vocab_mask, target_std=1.0)

    word2score = {w: float(s) for w, s in zip(vocab, calibrated)}

    # Sanity: print the seed word scores after calibration
    print("\n[seed scores after calibration] (should be near +/-1)")
    for w in POS_SEEDS + NEG_SEEDS:
        print(f"  {w}: {word2score.get(w, 'NA')}")

    # Print top positive / negative words
    sorted_w = sorted(word2score.items(), key=lambda kv: kv[1], reverse=True)
    print("\n[top +words]    ", [(w, round(s, 2)) for w, s in sorted_w[:15]])
    print("[top -words]    ", [(w, round(s, 2)) for w, s in sorted_w[-15:]])

    # ----- 5.6 Score each document -----
    print("\n[docs] scoring documents ...")
    scored_rows = []
    for r, toks in zip(corpus, docs_tokens):
        s = doc_score(toks, word2score)
        scored_rows.append({
            "doc_idx":  corpus.index(r),
            "media":    r.get("\ufeffmedia", r.get("media", "")),
            "date":     r.get("date", ""),
            "title":    r.get("title", ""),
            "url":      r.get("url", ""),
            "Score_S":  s,
            "n_tokens": len(toks),
        })

    # ----- 5.7 Save outputs -----
    out_csv = os.path.join(OUT_DIR, "semantic_scaling_scores_full.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "doc_idx", "media", "date", "title", "url", "Score_S", "n_tokens",
        ])
        w.writeheader()
        for r in scored_rows:
            w.writerow(r)
    print(f"\n[save] wrote {len(scored_rows)} docs -> {out_csv}")

    # Vocabulary scores (useful for QA / appendix)
    vocab_path = os.path.join(OUT_DIR, "semantic_vocab_scores.csv")
    with open(vocab_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word", "score_raw", "score_calibrated", "is_seed"])
        raw_list = raw_scores.tolist()
        cal_list = calibrated.tolist()
        seed_set = set(POS_SEEDS + NEG_SEEDS)
        for w_, rs, cs in zip(vocab, raw_list, cal_list):
            w.writerow([w_, round(rs, 4), round(cs, 4), int(w_ in seed_set)])
    print(f"[save] wrote vocab scores -> {vocab_path}")

    # ----- 5.8 Save axis metadata -----
    meta = {
        "embed_model": EMBED_MODEL,
        "pos_seeds": POS_SEEDS,
        "neg_seeds": NEG_SEEDS,
        "min_freq": MIN_FREQ,
        "n_vocab_raw": len(vocab_counter),
        "n_vocab_kept": len(vocab),
        "seed_scores_calibrated": {w: word2score[w] for w in POS_SEEDS + NEG_SEEDS},
    }
    with open(os.path.join(OUT_DIR, "semantic_scaling_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] wrote metadata -> semantic_scaling_meta.json")

    return scored_rows


if __name__ == "__main__":
    main()