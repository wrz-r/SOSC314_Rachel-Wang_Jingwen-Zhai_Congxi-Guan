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



