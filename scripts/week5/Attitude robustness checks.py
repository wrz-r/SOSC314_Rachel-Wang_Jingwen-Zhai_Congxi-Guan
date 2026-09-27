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



