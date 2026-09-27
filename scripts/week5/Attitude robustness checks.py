# Install and import packages
import importlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

for package, module in [("sentence-transformers", "sentence_transformers"), ("jieba", "jieba")]:
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from google.colab import files
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


# User-editable settings
OUTPUT_DIR = Path("/content/semantic_scaling_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep this fixed across every robustness run.
# This is the model specified in the uploaded semantic_scaling.py.
EMBED_MODEL = "shibing624/text2vec-base-chinese"
EMBED_BATCH_SIZE = 64

POS_SEEDS = ["幸福", "美满", "和谐", "陪伴", "稳定"]
NEG_SEEDS = ["压力", "负担", "痛苦", "束缚", "恐惧"]

BASELINE_MIN_FREQ = 3
MIN_FREQ_VALUES = [2, 3, 5]
AGGREGATION_VALUES = ["mean", "median", "trimmed_mean"]

STOPWORDS = set("""
的 了 在 是 我 你 他 她 它 我们 你们 他们 和 与 及 或 也 都 就 还
这 那 此 其 之 于 以 对 把 被 让 使 给 向 从 到 为 而 但 然而
一个 一些 这 那 一 不 没 没有 很 太 非常 比较 更 最 都 也 还 已经
将 把 将 word 并 并且 或者 而且 因为 所以 如果 虽然 但是 然后 因为
""".split())


# Upload the exact official-media and Weibo corpus files separately
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


OFFICIAL_FILE = upload_one_csv(
    "Upload the exact official-media CSV used in the original attitude analysis"
)
WEIBO_FILE = upload_one_csv(
    "Upload the exact final cleaned Weibo CSV used in the attitude analysis"
)


# Load, standardize, and tokenize both corpora
def read_csv_flexible(path):
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Could not find {description}. Tried {candidates}. "
        f"Available columns: {df.columns.tolist()}"
    )


def extract_year(value, fallback=""):
    match = re.search(r"20\d{2}", str(value))
    if match:
        return match.group()
    match = re.search(r"20\d{2}", str(fallback))
    return match.group() if match else ""


def tokenize(text):
    if not text:
        return []
    tokens = [token.strip() for token in jieba.cut(str(text))]
    return [
        token for token in tokens
        if token
        and token not in STOPWORDS
        and not re.fullmatch(r"[\W_]+", token)
        and len(token) >= 2
    ]


def standardize_corpus(path, source):
    df = read_csv_flexible(path)
    if source == "official":
        text_col = find_column(df, ["text", "clean_text", "正文", "文章正文"], "official text column")
        date_col = find_column(df, ["date", "发布时间", "publication_date"], "official date column")
        id_col = next((c for c in ["url", "id", "doc_idx"] if c in df.columns), None)
        fallback_year_col = "year" if "year" in df.columns else None
    else:
        text_col = find_column(df, ["微博正文", "clean_text", "text"], "Weibo text column")
        date_col = find_column(df, ["发布时间", "date"], "Weibo date column")
        id_col = next((c for c in ["id", "url", "doc_idx"] if c in df.columns), None)
        fallback_year_col = next((c for c in ["sampling_year", "year"] if c in df.columns), None)

    # Preserve the uploaded file's row index before assigning scalar metadata.
    out = pd.DataFrame(index=df.index)
    out["source"] = source
    out["doc_id"] = (
        df[id_col].fillna("").astype(str)
        if id_col else pd.Series([f"{source}_{i}" for i in range(len(df))])
    )
    out["text"] = df[text_col].fillna("").astype(str).str.strip()
    fallback = df[fallback_year_col] if fallback_year_col else pd.Series([""] * len(df))
    out["year"] = [extract_year(date, fb) for date, fb in zip(df[date_col], fallback)]
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    print(f"{source}: loaded {len(out):,} non-empty documents")
    return out


official = standardize_corpus(OFFICIAL_FILE, "official")
weibo = standardize_corpus(WEIBO_FILE, "weibo")
docs = pd.concat([official, weibo], ignore_index=True)

print("Tokenizing both corpora...")
docs["tokens"] = docs["text"].map(tokenize)
docs = docs[docs["tokens"].map(len) > 0].reset_index(drop=True)
print(f"Combined tokenized corpus: {len(docs):,} documents/posts")



# Embed the shared vocabulary and semantic anchors once
counter = Counter()
for tokens in docs["tokens"]:
    counter.update(tokens)

vocabulary = sorted([word for word, count in counter.items() if count >= min(MIN_FREQ_VALUES)])
word_to_index = {word: index for index, word in enumerate(vocabulary)}
word_frequency = np.asarray([counter[word] for word in vocabulary])

# Preserve repeated tokens, matching the original mean-of-token-scores approach.
doc_token_indices = [
    np.asarray([word_to_index[token] for token in tokens if token in word_to_index], dtype=np.int32)
    for tokens in docs["tokens"]
]

print(f"Shared vocabulary at MIN_FREQ >= {min(MIN_FREQ_VALUES)}: {len(vocabulary):,} words")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding model on {device}: {EMBED_MODEL}")
embedding_model = SentenceTransformer(EMBED_MODEL, device=device)

all_seed_words = POS_SEEDS + NEG_SEEDS
texts_to_embed = all_seed_words + vocabulary
all_embeddings = embedding_model.encode(
    texts_to_embed,
    batch_size=EMBED_BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)
seed_embeddings = {
    word: all_embeddings[index]
    for index, word in enumerate(all_seed_words)
}
vocab_embeddings = all_embeddings[len(all_seed_words):]
del all_embeddings


