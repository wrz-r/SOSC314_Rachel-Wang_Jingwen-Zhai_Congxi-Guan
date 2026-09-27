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



