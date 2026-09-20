"""
============================================================================
 Transformer training & attitude score prediction
============================================================================

Train a Chinese Transformer classifier on your manually labelled 114 texts
(three classes: +1 / 0 / -1), then use the trained model to score **all
3,935 official-media articles**. For each article we get:

    P(+1), P(0), P(-1)            -> probabilities over the three classes
    Score_T = P(+1) - P(-1)       -> continuous attitude score, range [-1, +1]

We also report validation accuracy / macro-F1 / confusion matrix, and dump
everything to disk so the next stage (Semantic Scaling) can compare against
Score_T.

============================================================================
"""

# =============================================================================
# Part 0 — Imports & global config
# =============================================================================
import os
import csv
import json
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

# ---------- Global config ----------

CORPUS       = "/Users/congxi/model/official_media_articles_2021_2025_merged.csv"
LABELLED     = "/Users/congxi/model/attitude_analysis/output/labelled_usable.csv"
OUT_DIR      = "/Users/congxi/model/attitude_analysis/output"

MODEL_NAME   = "hfl/chinese-roberta-wwm-ext"   # Chinese RoBERTa; stable on CPU
MAX_LEN      = 256                            # truncate to 256 tokens (speed)
BATCH_SIZE   = 8                              # CPU memory/speed tradeoff
EPOCHS       = 5                              # small data -> few epochs
LR           = 2e-5                           # classic BERT fine-tune LR
WARMUP_RATIO = 0.1
SEED         = 42                             # reproducibility

LABELS       = ["-1", "0", "+1"]               # label ordinal order
LABEL2ID     = {l: i for i, l in enumerate(LABELS)}
ID2LABEL     = {i: l for l, i in LABEL2ID.items()}

# Lock all randomness so the run is reproducible.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# Part 1 — Load labelled set & split into train / val
# =============================================================================
def load_labelled():
    """Read your 114 usable labelled rows."""
    with open(LABELLED, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[load_labelled] {len(rows)} usable labelled rows")
    for r in rows:
        r["attitude"] = r["attitude"].strip()
        r["label_id"] = LABEL2ID[r["attitude"]]
        r["text"] = (r["text"] or "").strip()
    return rows


def stratified_split(rows, test_size=0.2):
    """
    Stratified split by attitude label so every class survives in val.
    """
    train, val = train_test_split(
        rows, test_size=test_size,
        stratify=[r["label_id"] for r in rows],
        random_state=SEED,
    )
    print(f"[split] train={len(train)}, val={len(val)}")
    print(f"        train dist: {Counter(r['attitude'] for r in train)}")
    print(f"        val   dist: {Counter(r['attitude'] for r in val)}")
    return train, val


# =============================================================================
# Part 2 — PyTorch Dataset & DataLoader
# =============================================================================
class TextDataset(Dataset):
    """Wrap each (text, label) into a tensor dict the model can ingest."""
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(
            r["text"],
            max_length=MAX_LEN,
            padding="max_length",     # pad everything to MAX_LEN (simplest on CPU)
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(r["label_id"], dtype=torch.long),
        }


def make_loader(rows, tokenizer, shuffle):
    ds = TextDataset(rows, tokenizer)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# =============================================================================
# Part 3 — Train one epoch
# =============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()

        # Gradient clipping to stabilise Transformer fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(loader)


# =============================================================================
# Part 4 — Evaluate on the validation set
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, golds, probs_all = [], [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        probs = F.softmax(logits, dim=-1)         # -> probabilities
        pred = logits.argmax(dim=-1)

        preds.extend(pred.cpu().tolist())
        golds.extend(labels.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    acc = accuracy_score(golds, preds)
    f1m = f1_score(golds, preds, average="macro")  # macro-F1 fairer on imbalance
    cm  = confusion_matrix(golds, preds, labels=list(range(3)))
    return acc, f1m, cm, preds, golds, probs_all


# =============================================================================
# Part 5 — Predict on the full 3,935-article corpus
# =============================================================================
@torch.no_grad()
def predict_corpus(model, tokenizer, rows, device):
    """
    For each article return: P(-1), P(0), P(+1), Score_T = P(+1) - P(-1).
    """
    model.eval()
    results = []
    # Batched forward pass to avoid memory blow-up on 3,935 rows.
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        texts = [r.get("text", "") or "" for r in batch]
        enc = tokenizer(
            texts, max_length=MAX_LEN, padding=True,
            truncation=True, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = F.softmax(logits, dim=-1).cpu().tolist()

        for r, p in zip(batch, probs):
            p_neg, p_neu, p_pos = p[0], p[1], p[2]
            results.append({
                **r,
                "P_neg":   p_neg,
                "P_neu":   p_neu,
                "P_pos":   p_pos,
                "Score_T": p_pos - p_neg,         # continuous attitude score
            })
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  predict {i + len(batch)}/{len(rows)}")
    return results


# =============================================================================
# Part 6 — Main pipeline
# =============================================================================
def main():
    device = torch.device("cpu")  # no GPU available
    print(f"\n[device] {device}\n")

    # ------ Load data ------
    rows = load_labelled()
    train_rows, val_rows = stratified_split(rows, test_size=0.2)

    # ------ Load tokenizer & model ------
    print(f"\n[model] loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3,
        id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)
    print(f"[model] loaded. params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ------ DataLoader ------
    train_loader = make_loader(train_rows, tokenizer, shuffle=True)
    val_loader   = make_loader(val_rows,   tokenizer, shuffle=False)

    # ------ Optimizer + Scheduler ------
    optim = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # ------ Training loop ------
    print(f"\n[training] {EPOCHS} epochs x {len(train_loader)} steps = {total_steps} steps")
    best_f1 = -1.0
    best_state = None
    for ep in range(1, EPOCHS + 1):
        t0 = datetime.now()
        train_loss = train_one_epoch(model, train_loader, optim, sched, device)
        acc, f1m, cm, _, _, _ = evaluate(model, val_loader, device)
        dt = (datetime.now() - t0).total_seconds()
        print(f"\n[epoch {ep}] loss={train_loss:.4f} | val_acc={acc:.3f} | val_macroF1={f1m:.3f} | {dt:.0f}s")
        print("  confusion (rows=true, cols=pred):")
        print(f"    {LABELS}")
        for i, row in enumerate(cm):
            print(f"    {LABELS[i]} {row}")

        # Keep the best weights by macro-F1
        if f1m > best_f1:
            best_f1 = f1m
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  * new best macro-F1={f1m:.3f}")

    # Reload best weights
    if best_state:
        model.load_state_dict(best_state)
        print(f"\n[best] loaded best weights (val macro-F1={best_f1:.3f})")

    # ------ Final eval (best on val) ------
    acc, f1m, cm, val_preds, val_golds, val_probs = evaluate(model, val_loader, device)
    print("\n========================================")
    print(f"[final val] accuracy={acc:.3f}  macro-F1={f1m:.3f}")
    print("classification report:")
    print(classification_report(val_golds, val_preds, target_names=LABELS, digits=3))
    print("confusion matrix:")
    print(cm)
    print("========================================\n")

    # ------ Score the full corpus ------
    print("[predict] loading full corpus ...")
    with open(CORPUS, encoding="utf-8") as f:
        corpus = list(csv.DictReader(f))
    corpus_for_pred = []
    for i, r in enumerate(corpus):
        corpus_for_pred.append({
            "doc_idx": i,
            "media":   r.get("\ufeffmedia", r.get("media", "")),
            "date":    r.get("date", ""),
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "text":    (r.get("text", "") or "").strip(),
        })
    scored = predict_corpus(model, tokenizer, corpus_for_pred, device)

    # ------ Write outputs ------
    out_csv = os.path.join(OUT_DIR, "transformer_scores_full.csv")
    fields = ["doc_idx", "media", "date", "title", "url",
              "P_neg", "P_neu", "P_pos", "Score_T"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in scored:
            w.writerow(r)
    print(f"\n[save] wrote {len(scored)} scored docs -> {out_csv}")

    # Val predictions + gold, for manual inspection
    val_out = os.path.join(OUT_DIR, "transformer_val_predictions.csv")
    with open(val_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ann_id", "true_label", "pred_label",
                    "P_neg", "P_neu", "P_pos", "Score_T", "title"])
        for r, g, p, pr in zip(val_rows, val_golds, val_preds, val_probs):
            w.writerow([
                r["ann_id"], ID2LABEL[g], ID2LABEL[p],
                f"{pr[0]:.4f}", f"{pr[1]:.4f}", f"{pr[2]:.4f}",
                f"{pr[2] - pr[0]:.4f}",
                r["title"][:60],
            ])
    print(f"[save] wrote val predictions -> {val_out}")

    # Metrics summary
    metrics_path = os.path.join(OUT_DIR, "transformer_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "epochs": EPOCHS,
            "best_macro_f1": best_f1,
            "final_val_accuracy": acc,
            "final_val_macro_f1": f1m,
            "confusion_matrix": cm.tolist(),
            "labels": LABELS,
            "seed": SEED,
        }, f, indent=2)
    print(f"[save] wrote metrics -> {metrics_path}")


if __name__ == "__main__":
    main()


    """
============================================================================
 STEP 2 - Score EVERY document (official media + Weibo) with the Transformer
============================================================================

"""

import os
import re
import csv
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE      = "/Users/congxi/model/attitude_analysis"
MODEL_DIR = os.path.join(BASE, "model_attitude")
OUT_DIR   = os.path.join(BASE, "output")

OFFICIAL_CSV = "/Users/congxi/model/data_all/official_media_articles_2021_2025_merged.csv"
WEIBO_CSV    = "/Users/congxi/model/data_all/2021-2025_weibo.csv"
OUT_CSV      = os.path.join(OUT_DIR, "transformer_scores_combined.csv")

OFFICIAL_MAXLEN = 256     # long news articles
WEIBO_MAXLEN    = 128     # short social posts
BATCH_SIZE      = 32      # inference batch (bigger than training: no grads)

torch.set_num_threads(os.cpu_count() or 4)


# ==========================================================================
# Text cleaning
# ==========================================================================
URL_RE  = re.compile(r"https?://\S+")
MENTION = re.compile(r"@[\w\u4e00-\u9fff\-]+")
VIDEO   = re.compile(r"L[\w\u4e00-\u9fff]+的微博视频")


def clean_weibo(t: str) -> str:
    """Light normalisation so Weibo slang still reaches the model intact."""
    if not t:
        return ""
    t = URL_RE.sub(" ", t)
    t = t.replace("网页链接", " ")
    t = VIDEO.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = t.replace("#", " ")          # drop markers, keep hashtag words
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def parse_year(datestr: str, fallback: str = "") -> str:
    """Official dates are YYYY-MM-DD; Weibo dates are M/D/YYYY H:M."""
    if datestr:
        m = re.search(r"(20\d{2})", datestr)
        if m:
            return m.group(1)
    return str(fallback)[:4]


# ==========================================================================
# Core: batched, length-sorted inference
# ==========================================================================
@torch.no_grad()
def predict_texts(model, tokenizer, texts, max_len, device, tag=""):
    """
    Return a list of [P_neg, P_neu, P_pos] aligned with `texts` order.

    We sort by character length and process in sorted order so that each batch
    contains similarly-sized sequences -> far less padding -> faster on CPU.
    """
    n = len(texts)
    probs = [[1/3, 1/3, 1/3]] * n          # default for empty docs
    order = sorted(range(n), key=lambda i: len(texts[i]))

    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        idx    = order[start:start + BATCH_SIZE]
        batch  = [texts[j] if texts[j] else "空" for j in idx]
        enc = tokenizer(batch, max_length=max_len, padding=True,
                        truncation=True, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        p = F.softmax(logits, dim=-1).cpu().tolist()
        for j, pj in zip(idx, p):
            probs[j] = pj

        if (start // BATCH_SIZE) % 50 == 0:
            done = min(start + BATCH_SIZE, n)
            rate = done / max(time.time() - t0, 1e-6)
            eta  = (n - done) / max(rate, 1e-6)
            print(f"  [{tag}] {done}/{n}  ({rate:.1f} doc/s, ETA {eta/60:.1f} min)",
                  flush=True)
    return probs


# ==========================================================================
# Loaders -> list of dicts with normalised fields
# ==========================================================================
def load_official(limit=None):
    with open(OFFICIAL_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        out.append({
            "doc_idx": i,
            "source":  "official",
            "media":   r.get("media", ""),
            "date":    r.get("date", ""),
            "year":    parse_year(r.get("date", "")),
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "text":    (r.get("text", "") or "").strip(),
        })
    return out[:limit] if limit else out


def load_weibo(limit=None):
    with open(WEIBO_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        raw = (r.get("微博正文", "") or "").strip()
        out.append({
            "doc_idx": i,
            "source":  "weibo",
            "media":   "Weibo",
            "date":    r.get("发布时间", ""),
            "year":    parse_year(r.get("发布时间", ""), r.get("sampling_year", "")),
            "title":   "",                                   # Weibo has no title
            "url":     f"https://weibo.com/{r.get('user_id','')}/{r.get('id','')}",
            "text":    clean_weibo(raw),
        })
    # drop rows with empty text (nothing to score)
    out = [r for r in out if r["text"]]
    return out[:limit] if limit else out


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows per corpus (for a quick smoke test)")
    args = ap.parse_args()
    limit = args.limit or None

    device = torch.device("cpu")
    print(f"[device] cpu | threads={torch.get_num_threads()}")

    print(f"[model] loading saved model from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("[load] official corpus ...")
    official = load_official(limit)
    print(f"       {len(official)} docs")
    print("[load] weibo corpus ...")
    weibo = load_weibo(limit)
    print(f"       {len(weibo)} docs")

    # ----- Official -----
    print(f"\n[predict] official (max_len={OFFICIAL_MAXLEN}) ...")
    off_probs = predict_texts(model, tokenizer,
                              [r["text"] for r in official],
                              OFFICIAL_MAXLEN, device, tag="official")

    # ----- Weibo -----
    print(f"\n[predict] weibo (max_len={WEIBO_MAXLEN}) ...")
    wb_probs = predict_texts(model, tokenizer,
                             [r["text"] for r in weibo],
                             WEIBO_MAXLEN, device, tag="weibo")

    # ----- Merge & write -----
    rows_out = []
    for r, p in zip(official + weibo, off_probs + wb_probs):
        p_neg, p_neu, p_pos = p
        rows_out.append({
            "doc_idx": r["doc_idx"], "source": r["source"],
            "media": r["media"], "date": r["date"], "year": r["year"],
            "title": r["title"], "url": r["url"],
            "P_neg": round(p_neg, 6), "P_neu": round(p_neu, 6), "P_pos": round(p_pos, 6),
            "Score_T": round(p_pos - p_neg, 6),
        })

    fields = ["doc_idx", "source", "media", "date", "year", "title", "url",
              "P_neg", "P_neu", "P_pos", "Score_T"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n[save] {len(rows_out)} scored docs -> {OUT_CSV}")

    # Small summary so the log is self-documenting
    import statistics as st
    for src in ("official", "weibo"):
        vals = [r["Score_T"] for r in rows_out if r["source"] == src]
        if vals:
            print(f"  [{src}] n={len(vals)}  mean={st.mean(vals):+.3f}  "
                  f"sd={st.pstdev(vals):.3f}  min={min(vals):+.3f}  max={max(vals):+.3f}")

    # Record run metadata
    with open(os.path.join(OUT_DIR, "predict_meta.json"), "w") as f:
        json.dump({
            "model_dir": MODEL_DIR,
            "n_official": len(official),
            "n_weibo": len(weibo),
            "official_maxlen": OFFICIAL_MAXLEN,
            "weibo_maxlen": WEIBO_MAXLEN,
            "run_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    print("[save] predict_meta.json")
    print("[done] prediction finished.")


if __name__ == "__main__":
    main()

"""
============================================================================
 STEP 3 - Figures: Official Media vs User-Initiated Posts (marriage attitude)
============================================================================

"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

# --------------------------------------------------------------------------
# Paths & palette (the project's colour card)
# --------------------------------------------------------------------------
OUT_DIR = "/Users/congxi/model/attitude_analysis/output"
SCORES  = os.path.join(OUT_DIR, "transformer_scores_combined.csv")

C_OFFICIAL = "#4A7BB7"   # deep blue   -> Official Media
C_USER     = "#D9412B"   # deep red    -> User-Initiated (Weibo)
C_NEG      = "#D9412B"   # deep red    -> Negative
C_NEU      = "#FED081"   # gold        -> Neutral
C_POS      = "#4A7BB7"   # deep blue   -> Positive
INK        = "#333333"
GRID       = "#DDDDDD"

SRC_LABEL = {"official": "Official Media", "weibo": "User-Initiated Posts"}
SRC_COLOR = {"official": C_OFFICIAL, "weibo": C_USER}
CAT_LABEL = ["Negative", "Neutral", "Positive"]
CAT_COLOR = [C_NEG, C_NEU, C_POS]

# Publication-style global rc settings (English labels only)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load():
    df = pd.read_csv(SCORES)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].isin([2021, 2022, 2023, 2024, 2025])].copy()
    df["year"] = df["year"].astype(int)
    # Predicted category = argmax of the three probabilities
    prob = df[["P_neg", "P_neu", "P_pos"]].to_numpy()
    df["pred_cat"] = prob.argmax(axis=1)          # 0=Neg, 1=Neu, 2=Pos
    return df


def ci95(series):
    """95% CI half-width from the standard error of the mean."""
    n = series.count()
    if n < 2:
        return 0.0
    return float(1.96 * series.std(ddof=1) / np.sqrt(n))


# ==========================================================================
# Fig 1 - Violin + boxplot of Score_T by source
# ==========================================================================
def fig1_violin(df, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sources = ["official", "weibo"]
    data    = [df.loc[df["source"] == s, "Score_T"].to_numpy() for s in sources]
    positions = [1, 0]                       # official on top

    # Horizontal violins
    parts = ax.violinplot(data, positions=positions, orientation="horizontal",
                          showextrema=False, widths=0.85)
    for body, s in zip(parts["bodies"], sources):
        body.set_facecolor(SRC_COLOR[s])
        body.set_alpha(0.35)
        body.set_edgecolor(SRC_COLOR[s])
        body.set_linewidth(1.2)

    # Overlaid boxplots
    bp = ax.boxplot(data, positions=positions, orientation="horizontal", widths=0.14,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color=INK),
                    capprops=dict(color=INK))
    for patch, s in zip(bp["boxes"], sources):
        patch.set_facecolor(SRC_COLOR[s]); patch.set_alpha(0.9)

    # Mean markers
    for s, pos in zip(sources, positions):
        m = df.loc[df["source"] == s, "Score_T"].mean()
        ax.plot(m, pos, marker="D", color="white", markersize=6,
                markeredgecolor=INK, markeredgewidth=1.0, zorder=5)

    # Mann-Whitney U (two-sided) - non-parametric test of a shift
    u, p = mannwhitneyu(data[0], data[1], alternative="two-sided")
    ax.set_yticks(positions)
    ax.set_yticklabels([SRC_LABEL[s] for s in sources])
    ax.set_xlabel("Transformer attitude score  (Score_T = P(+1) − P(−1))")
    ax.set_title("Figure 1. Distribution of marriage attitudes by source")
    ax.set_xlim(-1, 1)
    ax.axvline(0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_axisbelow(True)

    # Mean/median annotation
    for s, pos in zip(sources, positions):
        v = df.loc[df["source"] == s, "Score_T"]
        ax.text(0.98, pos + 0.14,
                f"mean={v.mean():+.3f}  median={v.median():+.3f}  n={len(v):,}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=10)
    ax.text(0.01, -0.28, f"Mann–Whitney U p = {p:.2e}",
            transform=ax.transAxes, fontsize=10, color="#666666")

    fig.savefig(path); plt.close(fig)
    return {"fig1_mannwhitney_p": float(p),
            "fig1_official_mean": float(data[0].mean()),
            "fig1_weibo_mean": float(data[1].mean())}


# ==========================================================================
# Fig 2 - Mean Score_T by year, two lines + 95% CI
# ==========================================================================
def fig2_trend(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in ["official", "weibo"]:
        g = df[df["source"] == s].groupby("year")["Score_T"]
        mean = g.mean()
        half = g.apply(ci95)
        ax.errorbar(mean.index, mean.values, yerr=half.values,
                    marker="o" if s == "official" else "s",
                    ms=7, lw=2.2, capsize=4, color=SRC_COLOR[s],
                    label=SRC_LABEL[s])
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean attitude score (Score_T)")
    ax.set_title("Figure 2. Marriage attitudes over time (2021–2025)")
    ax.set_xticks(sorted(df["year"].unique()))
    ax.legend(frameon=False)
    fig.savefig(path); plt.close(fig)


# ==========================================================================
# Fig 3 - 100% stacked bar of predicted categories
# ==========================================================================
def fig3_stacked(df, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sources = ["official", "weibo"]
    ypos = [1, 0]

    for s, y in zip(sources, ypos):
        sub = df[df["source"] == s]
        props = [np.mean(sub["pred_cat"] == k) for k in range(3)]
        left = 0.0
        for prop, color, lab in zip(props, CAT_COLOR, CAT_LABEL):
            ax.barh(y, prop, left=left, color=color, edgecolor="white", height=0.55)
            if prop > 0.05:
                ax.text(left + prop / 2, y, f"{prop*100:.0f}%",
                        ha="center", va="center", fontsize=10,
                        color=INK if lab == "Neutral" else "white")
            left += prop

    ax.set_yticks(ypos)
    ax.set_yticklabels([SRC_LABEL[s] for s in sources])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of documents")
    ax.set_title("Figure 3. Predicted attitude categories by source")
    ax.grid(False)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(CAT_COLOR, CAT_LABEL)],
              ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.45), frameon=False)
    fig.savefig(path); plt.close(fig)


# ==========================================================================
# Fig 4 - Density of Score_T by source
# ==========================================================================
def fig4_density(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 70)
    for s in ["official", "weibo"]:
        v = df.loc[df["source"] == s, "Score_T"]
        ax.hist(v, bins=bins, density=True, alpha=0.45,
                color=SRC_COLOR[s], label=SRC_LABEL[s])
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Transformer attitude score (Score_T)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 4. Density of attitude scores by source")
    ax.set_xlim(-1, 1)
    ax.legend(frameon=False)
    fig.savefig(path); plt.close(fig)


# ==========================================================================
# Fig 5 (supplementary) - Gap over time
# ==========================================================================
def fig5_gap(df, path):
    off = df[df["source"] == "official"].groupby("year")["Score_T"].mean()
    usr = df[df["source"] == "weibo"].groupby("year")["Score_T"].mean()
    gap = (off - usr).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(gap.index, gap.values, color="#F7834D", alpha=0.85, width=0.6)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap = mean(Official) − mean(User)")
    ax.set_title("Figure 5 (supplementary). Official−User attitude gap over time")
    ax.set_xticks(sorted(df["year"].unique()))
    fig.savefig(path); plt.close(fig)
    return {str(y): float(v) for y, v in gap.items()}


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("[load]", SCORES)
    df = load()
    print(f"       {len(df):,} docs "
          f"(official={sum(df.source=='official'):,}, weibo={sum(df.source=='weibo'):,})")

    summary = {"n_docs": int(len(df))}

    print("[fig1] violin + boxplot ...")
    summary.update(fig1_violin(df, os.path.join(OUT_DIR, "fig1_violin_by_source.png")))

    print("[fig2] trend by year ...")
    fig2_trend(df, os.path.join(OUT_DIR, "fig2_trend_by_year.png"))

    print("[fig3] stacked categories ...")
    fig3_stacked(df, os.path.join(OUT_DIR, "fig3_stacked_categories.png"))

    print("[fig4] density by source ...")
    fig4_density(df, os.path.join(OUT_DIR, "fig4_density_by_source.png"))

    print("[fig5] gap over time ...")
    summary["gap_by_year"] = fig5_gap(df, os.path.join(OUT_DIR, "fig5_gap_over_time.png"))

    # Per-source descriptive stats
    stats = {}
    for s in ["official", "weibo"]:
        v = df.loc[df["source"] == s, "Score_T"]
        sub = df[df["source"] == s]
        stats[s] = {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "median": float(v.median()),
            "sd": float(v.std(ddof=1)),
            "pct_negative": float(np.mean(sub["pred_cat"] == 0)),
            "pct_neutral":  float(np.mean(sub["pred_cat"] == 1)),
            "pct_positive": float(np.mean(sub["pred_cat"] == 2)),
        }
    summary["by_source"] = stats

    with open(os.path.join(OUT_DIR, "attitude_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[save] attitude_summary.json")
    print("[done] figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
