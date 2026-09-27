"""
 DIAGNOSTICS - Training stability of the attitude classifier
"""

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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = "/Users/congxi/model/attitude_analysis"
LABELLED  = os.path.join(BASE, "output", "labelled_usable.csv")
OUT_DIR   = os.path.join(BASE, "output")
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

MAX_LEN, BATCH_SIZE, EPOCHS, LR, WARMUP = 256, 8, 5, 2e-5, 0.1
SEEDS     = [42, 7, 123, 2024, 31337]
LABELS    = ["-1", "0", "+1"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}

INK = "#333333"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.7,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

torch.set_num_threads(os.cpu_count() or 4)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_labelled():
    with open(LABELLED, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["attitude"] = r["attitude"].strip()
        r["label_id"] = LABEL2ID[r["attitude"]]
        r["text"]     = (r["text"] or "").strip()
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.tok = rows, tokenizer
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], max_length=MAX_LEN, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(r["label_id"], dtype=torch.long)}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        preds.extend(logits.argmax(dim=-1).tolist())
        golds.extend(batch["label"].tolist())
    return (accuracy_score(golds, preds),
            f1_score(golds, preds, average="macro"), preds, golds)


def train_model(train_rows, val_rows, tokenizer, device, seed):
    """Train one model from scratch; return (model, val_acc, val_f1)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    train_loader = DataLoader(TextDataset(train_rows, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_rows, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=False)

    optim = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        optim, int(steps * WARMUP), steps)

    best_f1, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for batch in train_loader:
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["label"])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
        acc, f1m, _, _ = evaluate(model, val_loader)
        if f1m > best_f1:
            best_f1, best_state = f1m, {k: v.detach().clone()
                                        for k, v in model.state_dict().items()}
        print(f"    ep{ep+1}: val_acc={acc:.3f} val_F1={f1m:.3f}", flush=True)

    model.load_state_dict(best_state)
    acc, f1m, _, _ = evaluate(model, val_loader)
    return model, acc, f1m


# --------------------------------------------------------------------------
# Gap check: score a small fixed subsample from each seed's model
# --------------------------------------------------------------------------
GAP_SAMPLE = os.path.join(OUT_DIR, "diag_sensitivity_scores_V0_baseline.csv")


@torch.no_grad()
def score_gap_sample(model, tokenizer, device, max_batch=32):
    """Re-score the V0 sensitivity sample (official + weibo/yr) with `model`."""
    import pandas as pd
    # need raw text -> reload corpora the same way predict_combined does
    import re
    URL_RE  = re.compile(r"https?://\S+")
    MENTION = re.compile(r"@[\w\u4e00-\u9fff\-]+")
    VIDEO   = re.compile(r"L[\w\u4e00-\u9fff]+的微博视频")

    def clean_weibo(t):
        if not t: return ""
        t = URL_RE.sub(" ", t); t = t.replace("网页链接", " ")
        t = VIDEO.sub(" ", t);  t = MENTION.sub(" ", t)
        t = t.replace("#", " ")
        return re.sub(r"\s+", " ", t).strip()

    def parse_year(s, fb=""):
        if s:
            m = re.search(r"(20\d{2})", s)
            if m: return m.group(1)
        return str(fb)[:4]

    off_csv = "/Users/congxi/model/data_all/official_media_articles_2021_2025_merged.csv"
    wb_csv  = "/Users/congxi/model/data_all/2021-2025_weibo.csv"

    docs = []
    with open(off_csv, encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            t = (r.get("text", "") or "").strip()
            if t:
                docs.append({"source": "official",
                             "year": parse_year(r.get("date", "")), "text": t})
    with open(wb_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    wb = []
    for r in rows:
        raw = (r.get("微博正文", "") or "").strip()
        if not raw: continue
        y = parse_year(r.get("发布时间", ""), r.get("sampling_year", ""))
        if y in [str(y_) for y_ in [2021, 2022, 2023, 2024, 2025]]:
            wb.append({"source": "weibo", "year": y, "text": clean_weibo(raw)})
    rng = random.Random(42)
    for y in ["2021", "2022", "2023", "2024", "2025"]:
        pool = [d for d in wb if d["year"] == y]
        docs.extend(rng.sample(pool, min(300, len(pool))))

    model.eval()
    n = len(docs)
    order = sorted(range(n), key=lambda i: len(docs[i]["text"]))
    for start in range(0, n, max_batch):
        idx = order[start:start + max_batch]
        batch = [docs[j]["text"] or "空" for j in idx]
        enc = tokenizer(batch, max_length=256 if docs[idx[0]]["source"] == "official"
                        else 128, padding=True, truncation=True,
                        return_tensors="pt")
        logits = model(**enc).logits
        p = F.softmax(logits, dim=-1).tolist()
        for j, pj in zip(idx, p):
            docs[j]["Score_T"] = pj[2] - pj[0]
        print(f"    gap-sample {min(start+max_batch,n)}/{n}", flush=True)

    df = pd.DataFrame(docs)
    out = {}
    for s in ["official", "weibo"]:
        out[f"mean_{s}"] = df[df.source == s]["Score_T"].mean()
    out["gap_overall"] = out["mean_official"] - out["mean_weibo"]
    for y in ["2021", "2022", "2023", "2024", "2025"]:
        o = df[(df.source == "official") & (df.year == y)]["Score_T"]
        w = df[(df.source == "weibo") & (df.year == y)]["Score_T"]
        out[f"gap_{y}"] = o.mean() - w.mean()
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    device = torch.device("cpu")
    print(f"[device] cpu | threads={torch.get_num_threads()}", flush=True)

    rows = load_labelled()
    print(f"[load] {len(rows)} labelled rows", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ================= A1: seed stability =================
    seed_rows = []
    for seed in SEEDS:
        print(f"\n===== A1 seed={seed} =====", flush=True)
        t0 = datetime.now()
        train_rows, val_rows = train_test_split(
            rows, test_size=0.2, stratify=[r["label_id"] for r in rows],
            random_state=seed)
        print(f"  train={len(train_rows)} val={len(val_rows)} "
              f"dist={Counter(r['attitude'] for r in train_rows)}", flush=True)
        model, acc, f1m = train_model(train_rows, val_rows, tokenizer, device, seed)
        print(f"  -> best val acc={acc:.3f}  macro-F1={f1m:.3f} "
              f"({(datetime.now()-t0).total_seconds():.0f}s)", flush=True)

        print("  [gap-check] scoring fixed subsample ...", flush=True)
        gap = score_gap_sample(model, tokenizer, device)
        seed_rows.append({
            "seed": seed, "val_acc": round(acc, 4),
            "val_macro_f1": round(f1m, 4),
            "mean_official": round(gap["mean_official"], 4),
            "mean_weibo": round(gap["mean_weibo"], 4),
            "gap_overall": round(gap["gap_overall"], 4),
            **{f"gap_{y}": round(gap[f"gap_{y}"], 4)
               for y in [2021, 2022, 2023, 2024, 2025]},
        })
        print(f"  -> gap_overall={gap['gap_overall']:+.4f}", flush=True)
        del model
        # intermediate save so partial results survive an interrupt
        import pandas as pd
        pd.DataFrame(seed_rows).to_csv(
            os.path.join(OUT_DIR, "diag_seed_stability.csv"), index=False)

    # ================= A2: 5-fold CV =================
    print(f"\n===== A2 5-fold stratified CV =====", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_all = [r["label_id"] for r in rows]
    cv_rows = []
    for k, (tr_idx, va_idx) in enumerate(skf.split(rows, y_all), 1):
        tr = [rows[i] for i in tr_idx]
        va = [rows[i] for i in va_idx]
        print(f"\n-- fold {k}: train={len(tr)} val={len(va)}", flush=True)
        t0 = datetime.now()
        model, acc, f1m = train_model(tr, va, tokenizer, device, seed=42 + k)
        cv_rows.append({"fold": k, "n_train": len(tr), "n_val": len(va),
                        "val_acc": round(acc, 4),
                        "val_macro_f1": round(f1m, 4)})
        print(f"  -> acc={acc:.3f} F1={f1m:.3f} "
              f"({(datetime.now()-t0).total_seconds():.0f}s)", flush=True)
        del model

    import pandas as pd
    sdf = pd.DataFrame(seed_rows)
    cdf = pd.DataFrame(cv_rows)
    cdf.to_csv(os.path.join(OUT_DIR, "diag_cv_results.csv"), index=False)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar([str(s) for s in sdf["seed"]], sdf["val_macro_f1"],
                color="#4A7BB7", alpha=0.85)
    axes[0].axhline(sdf["val_macro_f1"].mean(), color="#D9412B", ls="--", lw=1.5,
                    label=f"mean={sdf['val_macro_f1'].mean():.3f}")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Seed"); axes[0].set_ylabel("Val macro-F1")
    axes[0].set_title("A1. Val macro-F1 across seeds")
    axes[0].legend(frameon=False)

    years = [2021, 2022, 2023, 2024, 2025]
    for _, r in sdf.iterrows():
        axes[1].plot(years, [r[f"gap_{y}"] for y in years],
                     marker="o", ms=5, lw=1.5, alpha=0.7,
                     label=f"seed {r['seed']}")
    axes[1].axhline(0, color="#999999", lw=0.9, ls="--")
    axes[1].set_xticks(years)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Gap (Official − User)")
    axes[1].set_title("A1. Official–user gap across seeds")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_seed_stability.png"))
    plt.close(fig)
    print("[save] fig7_seed_stability.png")

    # ---- verdicts ----
    cv_mean = cdf["val_macro_f1"].mean()
    cv_sd   = cdf["val_macro_f1"].std(ddof=1)
    summary = {
        "A1_seed_stability": {
            "n_seeds": len(SEEDS),
            "f1_mean": float(sdf["val_macro_f1"].mean()),
            "f1_sd": float(sdf["val_macro_f1"].std(ddof=1)),
            "f1_range": [float(sdf["val_macro_f1"].min()),
                         float(sdf["val_macro_f1"].max())],
            "gap_overall_all_positive": bool((sdf["gap_overall"] > 0).all()),
            "gap_overall_range": [float(sdf["gap_overall"].min()),
                                  float(sdf["gap_overall"].max())],
        },
        "A2_cv": {
            "n_folds": 5,
            "f1_mean": float(cv_mean), "f1_sd": float(cv_sd),
            "acc_mean": float(cdf["val_acc"].mean()),
            "per_fold": cdf.to_dict(orient="records"),
        },
    }
    with open(os.path.join(OUT_DIR, "diag_training_stability.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
