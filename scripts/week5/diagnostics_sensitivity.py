"""
 DIAGNOSTIC - Sensitivity of results to preprocessing choices
"""

import os
import re
import csv
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
BASE      = "/Users/congxi/model/attitude_analysis"
MODEL_DIR = os.path.join(BASE, "model_attitude")
OUT_DIR   = os.path.join(BASE, "output")

OFFICIAL_CSV = "/Users/congxi/model/data_all/official_media_articles_2021_2025_merged.csv"
WEIBO_CSV    = "/Users/congxi/model/data_all/2021-2025_weibo.csv"

BATCH_SIZE = 32
SEED       = 42
WEIBO_PER_YEAR = 1500
YEARS      = [2021, 2022, 2023, 2024, 2025]

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
# Text cleaning variants
# --------------------------------------------------------------------------
URL_RE  = re.compile(r"https?://\S+")
MENTION = re.compile(r"@[\w\u4e00-\u9fff\-]+")
VIDEO   = re.compile(r"L[\w\u4e00-\u9fff]+的微博视频")


def clean_base(t: str) -> str:
    """V0/V3/V4: current pipeline."""
    if not t:
        return ""
    t = URL_RE.sub(" ", t)
    t = t.replace("网页链接", " ")
    t = VIDEO.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = t.replace("#", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean_raw(t: str) -> str:
    """V1: no cleaning at all."""
    return (t or "").strip()


def clean_keephash(t: str) -> str:
    """V2: same as base but keep '#' characters."""
    if not t:
        return ""
    t = URL_RE.sub(" ", t)
    t = t.replace("网页链接", " ")
    t = VIDEO.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


VARIANTS = {
    # name: (weibo_cleaner, official_maxlen, weibo_maxlen)
    "V0_baseline":  (clean_base,     256, 128),
    "V1_raw_text":   (clean_raw,     256, 128),
    "V2_keep_hash":  (clean_keephash, 256, 128),
    "V3_maxlen_128": (clean_base,    128, 128),
    "V4_maxlen_256": (clean_base,    256, 256),
}


def parse_year(datestr: str, fallback: str = "") -> str:
    if datestr:
        m = re.search(r"(20\d{2})", datestr)
        if m:
            return m.group(1)
    return str(fallback)[:4]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_official():
    with open(OFFICIAL_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        out.append({
            "doc_idx": f"off_{i}",
            "source": "official",
            "year": parse_year(r.get("date", "")),
            "text_raw": (r.get("text", "") or "").strip(),
        })
    return [r for r in out if r["text_raw"]]


def load_weibo_sample():
    """Stratified sample: 1,500 docs per year, same docs across variants."""
    with open(WEIBO_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    docs = []
    for i, r in enumerate(rows):
        raw = (r.get("微博正文", "") or "").strip()
        if not raw:
            continue
        y = parse_year(r.get("发布时间", ""), r.get("sampling_year", ""))
        if y in [str(y_) for y_ in YEARS]:
            docs.append({"doc_idx": f"wb_{i}", "source": "weibo",
                         "year": y, "text_raw": raw})
    rng = random.Random(SEED)
    sample = []
    for y in [str(y_) for y_ in YEARS]:
        pool = [d for d in docs if d["year"] == y]
        sample.extend(rng.sample(pool, min(WEIBO_PER_YEAR, len(pool))))
    return sample


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, tokenizer, texts, max_len, tag=""):
    n = len(texts)
    probs = [[1/3, 1/3, 1/3]] * n
    order = sorted(range(n), key=lambda i: len(texts[i]))
    import time
    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        batch = [texts[j] if texts[j] else "空" for j in idx]
        enc = tokenizer(batch, max_length=max_len, padding=True,
                        truncation=True, return_tensors="pt")
        logits = model(**{k: v for k, v in enc.items()}).logits
        p = F.softmax(logits, dim=-1).cpu().tolist()
        for j, pj in zip(idx, p):
            probs[j] = pj
        if (start // BATCH_SIZE) % 100 == 0:
            done = min(start + BATCH_SIZE, n)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"    [{tag}] {done}/{n} ({rate:.1f} doc/s)", flush=True)
    return probs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("[model] loading ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    print("[load] official corpus ...", flush=True)
    official = load_official()
    print(f"       {len(official)} docs")
    print("[load] weibo stratified sample ...", flush=True)
    weibo = load_weibo_sample()
    print(f"       {len(weibo)} docs "
          f"({WEIBO_PER_YEAR}/year, seed={SEED})")

    summary_rows = []

    for vname, (wclean, off_len, wb_len) in VARIANTS.items():
        print(f"\n===== {vname} (official maxlen={off_len}, weibo maxlen={wb_len}) =====",
              flush=True)

        off_texts = [r["text_raw"] for r in official]          # official: never re-cleaned
        wb_texts  = [wclean(r["text_raw"]) for r in weibo]
        wb_texts  = [t if t else r["text_raw"] for t, r in zip(wb_texts, weibo)]

        print(f"  [predict] official ...", flush=True)
        off_p = predict(model, tokenizer, off_texts, off_len, tag=f"{vname}-off")
        print(f"  [predict] weibo ...", flush=True)
        wb_p  = predict(model, tokenizer, wb_texts, wb_len, tag=f"{vname}-wb")

        rows = []
        for r, p in zip(official, off_p):
            rows.append({"doc_idx": r["doc_idx"], "source": "official",
                         "year": r["year"],
                         "P_neg": p[0], "P_neu": p[1], "P_pos": p[2],
                         "Score_T": p[2] - p[0]})
        for r, p in zip(weibo, wb_p):
            rows.append({"doc_idx": r["doc_idx"], "source": "weibo",
                         "year": r["year"],
                         "P_neg": p[0], "P_neu": p[1], "P_pos": p[2],
                         "Score_T": p[2] - p[0]})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, f"diag_sensitivity_scores_{vname}.csv"),
                  index=False)

        # per-variant means + yearly gaps
        m_off = df[df.source == "official"]["Score_T"].mean()
        m_wb  = df[df.source == "weibo"]["Score_T"].mean()
        row = {"variant": vname,
               "n_official": len(official), "n_weibo": len(weibo),
               "mean_official": round(m_off, 4),
               "mean_weibo": round(m_wb, 4),
               "gap_overall": round(m_off - m_wb, 4)}
        for y in YEARS:
            ys = str(y)
            o = df[(df.source == "official") & (df.year == ys)]["Score_T"]
            w = df[(df.source == "weibo") & (df.year == ys)]["Score_T"]
            row[f"gap_{y}"] = round(o.mean() - w.mean(), 4)
        summary_rows.append(row)
        print(f"  -> mean_official={m_off:+.4f}  mean_weibo={m_wb:+.4f}  "
              f"gap={m_off - m_wb:+.4f}", flush=True)

    # ---- summary table ----
    s = pd.DataFrame(summary_rows)
    s.to_csv(os.path.join(OUT_DIR, "diag_sensitivity_summary.csv"), index=False)
    print("\n[save] diag_sensitivity_summary.csv")
    print(s.to_string(index=False))

    # ---- figure: gap lines per variant ----
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    markers = ["o", "s", "^", "D", "v"]
    for i, r in s.iterrows():
        ax.plot(YEARS, [r[f"gap_{y}"] for y in YEARS],
                marker=markers[i % 5], lw=1.8, ms=6, label=r["variant"])
    ax.axhline(0, color="#999999", lw=0.9, ls="--")
    ax.set_xticks(YEARS)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap = mean(Official) − mean(User)")
    ax.set_title("Sensitivity of the official–user gap to preprocessing choices")
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, "fig6_sensitivity_gap.png"))
    plt.close(fig)
    print("[save] fig6_sensitivity_gap.png")

    # ---- verdict ----
    gaps_positive = all(
        (s[f"gap_{y}"] > 0).all() for y in YEARS
    )
    spread = {y: float(s[f"gap_{y}"].max() - s[f"gap_{y}"].min()) for y in YEARS}
    with open(os.path.join(OUT_DIR, "diag_sensitivity_meta.json"), "w") as f:
        json.dump({
            "variants": list(VARIANTS.keys()),
            "weibo_per_year": WEIBO_PER_YEAR,
            "seed": SEED,
            "all_variants_all_years_gap_positive": gaps_positive,
            "max_min_spread_by_year": spread,
            "run_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    print(f"\n[verdict] gap positive in every variant × every year: {gaps_positive}")
    print(f"[verdict] largest cross-variant spread per year: "
          f"{ {y: round(v,4) for y, v in spread.items()} }")


if __name__ == "__main__":
    main()
