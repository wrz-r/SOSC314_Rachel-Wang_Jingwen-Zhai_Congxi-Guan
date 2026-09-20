# -*- coding: utf-8 -*-
"""
PHASE 3 — Attitude analysis: how is marriage framed in official media?

Pipeline
  Step 1  audit the human-coded pilot (attitude_coding_sample_50.csv)
  Step 2  Method A  dictionary baseline (codebook lexicon)
  Step 3  Method B  supervised classifier (TF-IDF char n-grams + logistic)
  Step 4  Method C  LSS-style semantic scaling (corpus-based polarity axis)
  Step 5  validation of all three against the human labels
  Step 6  agreement between the three operationalisations
  Step 7  score the full corpus -> attitude_score in [-1, +1]
  Step 8  downstream: attitude by year / topic / media / marriage-related
  Step 9  figures + tables (palette of the user's colour card, bilingual)

"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attitude_lexicon import (build_lexicon, CODEBOOK_MD, NON_TARGET_MARKERS,
                              POS_TH, NEG_TH, SEED_POS, SEED_NEG)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- colour card ----------
C_POS, C_NEU, C_NEG = "#4A7BB7", "#FED081", "#D9412B"
C_METHODS = ["#D9412B", "#4A7BB7", "#F7834D"]     # A, B, C
GRID = "#DDDDDD"
INK = "#333333"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # /Users/congxi/model
ADIR = os.path.join(BASE, "attitude_analysis")
OUT = os.path.join(ADIR, "output")
os.makedirs(OUT, exist_ok=True)

CORPUS = os.path.join(BASE, "text_as_data_output", "processed_corpus.csv")
LABELS = os.path.join(BASE, "attitude_coding_sample_50.csv")
TOPIC_DIST = os.path.join(BASE, "text_as_data_output", "topic_model",
                          "document_topic_distribution.csv")
SEED = 42

# ============================================================
# STEP 1. Load corpus + audited human labels
# ============================================================
corpus = pd.read_csv(CORPUS)
corpus["url"] = corpus["url"].astype(str)
corpus["clean_text"] = corpus["clean_text"].fillna("").astype(str)

lab = pd.read_csv(LABELS)
lab["url"] = lab["url"].astype(str)
lab["attitude_raw"] = lab["attitude"].astype(str).str.strip()

# normalise the coding vocabulary: 1/0/-1 keep, U and NR are excluded from training
def to_label(x):
    if x in ("1", "1.0", "+1"):
        return 1
    if x in ("-1", "-1.0"):
        return -1
    if x in ("0", "0.0"):
        return 0
    return np.nan          # U, NR

lab["y"] = lab["attitude_raw"].map(to_label)
audit = lab["attitude_raw"].value_counts(dropna=False)
print("=" * 60)
print("STEP 1  human-coded pilot audit")
print(audit.to_dict())

merged = lab.merge(corpus[["url", "clean_text", "tokens_clean", "year", "media",
                           "relevance", "marriage_related"]],
                   on="url", how="left", suffixes=("", "_c"))
print(f"matched to corpus: {merged['clean_text'].notna().sum()}/{len(merged)}")

train = merged.dropna(subset=["y"]).copy()
train["y"] = train["y"].astype(int)
counts = train["y"].value_counts().to_dict()
print(f"usable labelled docs: {len(train)}   by class: {counts}")
print("class balance note: negatives are severely under-represented "
      "-> supervised learning is a demonstration, not a finished model")

audit_df = pd.DataFrame({
    "item": ["coded_docs", "usable_+1/0/-1", "excluded_U", "excluded_NR",
             "class_+1", "class_0", "class_-1"],
    "value": [len(lab), len(train),
              int((lab["attitude_raw"] == "U").sum()),
              int((lab["attitude_raw"] == "NR").sum()),
              counts.get(1, 0), counts.get(0, 0), counts.get(-1, 0)],
})
audit_df.to_csv(os.path.join(OUT, "step1_label_audit.csv"), index=False)
train.drop(columns=["clean_text_c"]).to_csv(
    os.path.join(OUT, "step1_labelled_training_docs.csv"), index=False)

# ============================================================
# STEP 2. Method A - dictionary baseline
#    score = tanh( (pos_weight - neg_weight) / 1.5 ), matches on raw text so
#    multi-character phrases (高额彩礼, 白头偕老) are captured
# ============================================================
LEX = build_lexicon()
POS_ITEMS = [(w, v) for w, v in LEX.items() if v > 0]
NEG_ITEMS = [(w, v) for w, v in LEX.items() if v < 0]

def count_hits(text, items):
    return sum(text.count(w) * wt for w, wt in items)

# Codebook Rule 1 ("target rule") implemented as a shrinkage factor:
# if the document is dominated by procedural / institutional / legal context,
# a positive word is more likely to praise a service, policy or court ruling
# than marriage itself -> pull the score towards 0.
OFF_TARGET = NON_TARGET_MARKERS + ["法院", "判决", "案件", "审理", "纠纷",
                                   "检察", "犯罪嫌疑人", "条例", "实施办法"]

def method_a(text):
    text = "" if not isinstance(text, str) else text
    pos = count_hits(text, POS_ITEMS)
    neg = count_hits(text, NEG_ITEMS)
    off = sum(text.count(w) for w in OFF_TARGET)
    shrink = 1.0 / (1.0 + 0.35 * off)      # >= 0, decreasing in off-target cues
    return float(np.tanh((pos - neg) / 1.5) * shrink), pos, neg

a_scores, a_pos, a_neg = [], [], []
for t in corpus["clean_text"]:
    s, p, n = method_a(t)
    a_scores.append(s); a_pos.append(p); a_neg.append(n)
corpus["A_dict"] = a_scores
corpus["A_pos_hits"] = a_pos
corpus["A_neg_hits"] = a_neg

# ============================================================
# STEP 3. Method B - supervised classifier
#    char 2-4 grams (robust to segmentation) + opinionated logistic regression
# ============================================================
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                      sublinear_tf=True)
X_all = vec.fit_transform(corpus["clean_text"])
row_of = {u: i for i, u in enumerate(corpus["url"])}
tr_rows = [row_of[u] for u in train["url"]]
X_tr = X_all[tr_rows]
y_tr = train["y"].values

clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000,
                         random_state=SEED)
clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_all)
classes = list(clf.classes_)
corpus["B_prob_pos"] = proba[:, classes.index(1)]
corpus["B_prob_neg"] = proba[:, classes.index(-1)] if -1 in classes else 0.0
corpus["B_score"] = corpus["B_prob_pos"] - corpus["B_prob_neg"]
print("=" * 60)
print(f"STEP 3  classifier trained on {len(train)} docs, "
      f"classes={classes}")

# --- validation 1: leave-one-out on the feasible binary task (pos vs rest)
bin_y = (y_tr == 1).astype(int)
loo = LeaveOneOut()
pred = np.zeros(len(bin_y), dtype=int)
for tr, te in loo.split(X_tr):
    if len(np.unique(bin_y[tr])) < 2:
        pred[te] = int(round(bin_y[tr].mean()))
        continue
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000,
                           random_state=SEED).fit(X_tr[tr], bin_y[tr])
    pred[te] = m.predict(X_tr[te])[0]
acc_bin = accuracy_score(bin_y, pred)
print(f"LOO binary (positive vs rest): acc={acc_bin:.3f}  "
      f"majority baseline={max(bin_y.mean(), 1-bin_y.mean()):.3f}")

# --- validation 2: leave-one-out 3-class predictions (honest, out-of-sample)
def cat_from_score(s):
    return np.where(s > POS_TH, 1, np.where(s < NEG_TH, -1, 0))

loo3 = np.zeros(len(y_tr), dtype=int)
for tr, te in loo.split(X_tr):
    present = np.unique(y_tr[tr])
    if len(present) == 1:
        loo3[te] = int(present[0])
        continue
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000,
                           random_state=SEED).fit(X_tr[tr], y_tr[tr])
    loo3[te] = int(m.predict(X_tr[te])[0])
kappa_b = cohen_kappa_score(y_tr, loo3)
f1_b = f1_score(y_tr, loo3, average="macro", labels=[-1, 0, 1],
                zero_division=0)
pred_b_insample = clf.predict(X_tr)
print(f"LOO 3-class: kappa={kappa_b:.3f} macroF1={f1_b:.3f} "
      f"acc={accuracy_score(y_tr, loo3):.3f} "
      f"(in-sample kappa={cohen_kappa_score(y_tr, pred_b_insample):.3f} "
      f"-> shows pure memorisation, ignore it)")
cm_b = confusion_matrix(y_tr, loo3, labels=[-1, 0, 1])

# ============================================================
# STEP 4. Method C - LSS-style semantic scaling
#    seeds define the two poles; every vocabulary word inherits an
#    orientation from the documents it appears in; documents are scored
#    on that axis. No human labels are used.
# ============================================================
def seed_polarity(text):
    pos = sum(text.count(w) for w in SEED_POS)
    neg = sum(text.count(w) for w in SEED_NEG)
    return (pos - neg) / (pos + neg + 1.0)

corpus["C_seed_polarity"] = corpus["clean_text"].map(seed_polarity)

# orientation of each vocabulary word = mean seed polarity of docs using it
ori_num, ori_cnt = {}, {}
for toks, pol in zip(corpus["tokens_clean"].fillna("").astype(str),
                     corpus["C_seed_polarity"]):
    for w in set(toks.split()):
        ori_num[w] = ori_num.get(w, 0.0) + pol
        ori_cnt[w] = ori_cnt.get(w, 0) + 1

orient = {w: ori_num[w] / ori_cnt[w]
          for w in ori_num if ori_cnt[w] >= 50 and abs(ori_num[w] / ori_cnt[w]) > 0}
weight = {w: o * min(1.0, np.log(ori_cnt[w]) / np.log(len(corpus)))
          for w, o in orient.items()}
print("=" * 60)
print(f"STEP 4  semantic axis built from {len(weight)} vocabulary words")
top_axis = sorted(weight.items(), key=lambda kv: kv[1])
print("  most negative axis words:", [w for w, _ in top_axis[:8]])
print("  most positive axis words:", [w for w, _ in top_axis[-8:]][::-1])

def method_c_raw(toks):
    ws = [w for w in toks.split() if w in weight]
    if not ws:
        return 0.0
    return float(np.mean([weight[w] for w in ws]))

raw_c = corpus["tokens_clean"].fillna("").astype(str).map(method_c_raw)
# unsupervised calibration of the zero point: the axis is centred on the
# corpus average and scaled by its own dispersion, so 0 = "average framing"
# (no human labels are used at this step)
corpus["C_scaling"] = np.tanh((raw_c - raw_c.mean()) / (raw_c.std() + 1e-9))

# ============================================================
# STEP 5. Validation of A and C against the human labels
# ============================================================
val_rows = []
train = train.merge(corpus[["url", "A_dict", "B_score", "C_scaling"]],
                    on="url", how="left")
for name, col in [("A_dict", "A_dict"), ("B_classifier", "B_score"),
                  ("C_scaling", "C_scaling")]:
    s = train[col].values
    rho = spearmanr(s, y_tr).statistic
    if name == "B_classifier":
        k, acc = kappa_b, accuracy_score(y_tr, loo3)
    else:
        k = cohen_kappa_score(y_tr, cat_from_score(s))
        acc = accuracy_score(y_tr, cat_from_score(s))
    val_rows.append({"method": name, "spearman_vs_human": round(rho, 3),
                     "cohen_kappa": round(k, 3), "accuracy": round(acc, 3)})
val = pd.DataFrame(val_rows)
val.to_csv(os.path.join(OUT, "step5_method_validation.csv"), index=False)
print("=" * 60)
print("STEP 5  validation against the human coding")
print(val.to_string(index=False))
print("confusion matrix of Method B (rows=true -1/0/+1):")
print(cm_b)
pd.DataFrame(cm_b, index=["true_-1", "true_0", "true_+1"],
             columns=["pred_-1", "pred_0", "pred_+1"]).to_csv(
    os.path.join(OUT, "step5_confusion_B.csv"))

# ============================================================
# STEP 6. Agreement between the three operationalisations
# ============================================================
agree = []
for c1, c2 in [("A_dict", "B_score"), ("A_dict", "C_scaling"),
               ("B_score", "C_scaling")]:
    rho = spearmanr(corpus[c1], corpus[c2]).statistic
    k = cohen_kappa_score(cat_from_score(corpus[c1]),
                          cat_from_score(corpus[c2]))
    agree.append({"pair": f"{c1} vs {c2}", "spearman_all_docs": round(rho, 3),
                  "cohen_kappa_cats": round(k, 3)})
agree_df = pd.DataFrame(agree)
agree_df.to_csv(os.path.join(OUT, "step6_method_agreement.csv"), index=False)
print("=" * 60)
print("STEP 6  agreement between methods (full corpus)")
print(agree_df.to_string(index=False))

# composite: average of the three standardised scores (robustness view)
z = corpus[["A_dict", "B_score", "C_scaling"]].apply(
    lambda s: (s - s.mean()) / (s.std() + 1e-9))
corpus["attitude_score"] = np.tanh(1.5 * z[["A_dict", "C_scaling"]].mean(axis=1))
corpus["attitude_score"] = corpus["attitude_score"].clip(-1, 1)
# categories for the continuous score: +/-0.30 keeps "neutral" meaningful
# (Method B is stored but excluded until it has enough training labels)
corpus["attitude_cat"] = np.where(corpus["attitude_score"] > 0.30, 1,
                                  np.where(corpus["attitude_score"] < -0.30, -1, 0))

corpus.to_csv(os.path.join(OUT, "step7_document_attitude_scores.csv"),
              index=False)
print("=" * 60)
print("STEP 7  full-corpus scores written "
      f"({len(corpus)} docs); cat distribution:",
      corpus["attitude_cat"].value_counts().to_dict())

# codebook v1 is written out as a project artefact
with open(os.path.join(ADIR, "codebook_v1.md"), "w", encoding="utf-8") as f:
    f.write(CODEBOOK_MD)

# ============================================================
# STEP 8. Downstream: attitude by year / topic / media / marriage flag
# ============================================================
main = corpus[corpus["relevance"] == "highly_relevant"].copy()
print(f"main analytic sample (highly_relevant): {len(main)}")

def mean_ci(s):
    s = pd.Series(s).dropna()
    if len(s) < 2:
        return s.mean(), np.nan
    return s.mean(), 1.96 * s.std(ddof=1) / np.sqrt(len(s))

# --- by year
rows = []
for y, g in main.groupby("year"):
    for name, col in [("A_dictionary", "A_dict"), ("B_classifier", "B_score"),
                      ("C_scaling", "C_scaling"),
                      ("consensus", "attitude_score")]:
        m, ci = mean_ci(g[col])
        rows.append({"year": y, "method": name, "mean": m, "ci95": ci, "n": len(g)})
by_year = pd.DataFrame(rows)
by_year.to_csv(os.path.join(OUT, "step8_attitude_by_year.csv"), index=False)

# --- join the LDA topic assignment (matched on media + date + title)
topics_df = pd.read_csv(TOPIC_DIST)
topics_df = topics_df.drop_duplicates(subset=["url"]).copy()
topics_df["url"] = topics_df["url"].astype(str)
main = main.merge(topics_df[["url", "dominant_topic"]], on="url", how="left")
main = main.drop_duplicates(subset=["url"])          # guard: one row per article
match_rate = main["dominant_topic"].notna().mean()
print(f"topic join success rate: {match_rate:.1%}  (rows={len(main)})")

TOPIC_EN = {
    1: "彩礼/婚俗改革 Bride price & custom reform",
    2: "青年发展/对外交流 Youth & national development",
    3: "离婚司法/涉婚案件 Divorce litigation",
    4: "年轻人婚恋观 Youth dating & marriage views",
    5: "婚姻登记/生育政策 Registration & fertility policy",
    6: "网页模板噪音 Web-page boilerplate",
}
by_topic = (main.dropna(subset=["dominant_topic"])
                .groupby("dominant_topic")
                .agg(mean_attitude=("attitude_score", "mean"),
                     mean_A=("A_dict", "mean"), mean_B=("B_score", "mean"),
                     mean_C=("C_scaling", "mean"), n=("url", "count"))
                .reset_index())
by_topic["topic_label"] = by_topic["dominant_topic"].map(TOPIC_EN)
by_topic.to_csv(os.path.join(OUT, "step8_attitude_by_topic.csv"), index=False)
print(by_topic.round(3).to_string(index=False))

# --- by media outlet
by_media = (main.groupby("media")
                .agg(mean_attitude=("attitude_score", "mean"),
                     n=("url", "count"))
                .query("n >= 50")
                .sort_values("mean_attitude"))
by_media.to_csv(os.path.join(OUT, "step8_attitude_by_media.csv"))

# --- marriage-related vs other
by_flag = (main.assign(flag=main["marriage_related"].astype(str))
               .groupby("flag")
               .agg(mean_attitude=("attitude_score", "mean"),
                    mean_A=("A_dict", "mean"), mean_C=("C_scaling", "mean"),
                    n=("url", "count")))
by_flag.to_csv(os.path.join(OUT, "step8_attitude_by_marriage_flag.csv"))
print(by_flag.round(3).to_string())

# ============================================================
# STEP 9. Figures 
# ============================================================
# (a) validation scatter: human label vs Methods A and C
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
COL = {1: C_POS, 0: C_NEU, -1: C_NEG}
for ax, (col, title) in zip(axes, [("A_dict", "方法A 词典 Dictionary"),
                                   ("C_scaling", "方法C 语义缩放 Semantic scaling")]):
    for lab_v in [1, 0, -1]:
        sub = train[train["y"] == lab_v]
        ax.scatter(np.full(len(sub), lab_v) + np.random.uniform(-.12, .12, len(sub)),
                   sub[col], s=55, color=COL[lab_v], alpha=.85,
                   edgecolor="white", label={1: "人工 +1", 0: "人工 0", -1: "人工 -1"}[lab_v])
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks([-1, 0, 1]); ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("人工标注 Human label", color=INK)
    ax.set_ylabel("模型得分 Model score", color=INK)
    ax.set_title(title, color=INK)
    ax.grid(axis="y", ls=":", color=GRID)
axes[0].legend(fontsize=9, frameon=False)
fig.suptitle("方法验证：模型得分 vs 人工标注 Validation on the 50-document pilot",
             color=INK, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, .95])
plt.savefig(os.path.join(OUT, "fig1_validation.png"), dpi=200)
plt.close()

# (b) attitude over time, one line per modelling choice
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, (name, ls) in enumerate([("A_dictionary", "--"), ("B_classifier", "-."),
                               ("C_scaling", ":")]):
    d = by_year[by_year["method"] == name].sort_values("year")
    ax.plot(d["year"], d["mean"], ls, color=C_METHODS[i], lw=1.8, marker="o",
            ms=5, label={"A_dictionary": "A 词典 Dictionary",
                         "B_classifier": "B 分类器 Classifier",
                         "C_scaling": "C 语义缩放 Semantic scaling"}[name])
d = by_year[by_year["method"] == "consensus"].sort_values("year")
ax.errorbar(d["year"], d["mean"], yerr=d["ci95"], color=INK, lw=3,
            marker="s", ms=7, capsize=4, label="综合 Consensus (95% CI)")
ax.axhline(0, color=GRID, lw=1)
ax.set_xticks(sorted(main["year"].unique()))
ax.set_xlabel("年份 Year", color=INK)
ax.set_ylabel("婚姻态度得分 Attitude toward marriage (−1 … +1)", color=INK)
ax.set_title("官方媒体对婚姻的态度 2021–2025 Attitude toward marriage, official media",
             color=INK, fontsize=13)
ax.grid(axis="y", ls=":", color=GRID)
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_attitude_by_year.png"), dpi=200)
plt.close()

# (c) attitude by topic
fig, ax = plt.subplots(figsize=(11, 5.5))
d = by_topic.sort_values("mean_attitude")
colors = [C_NEG if v < -0.02 else (C_POS if v > 0.02 else C_NEU)
          for v in d["mean_attitude"]]
ax.barh(range(len(d)), d["mean_attitude"], color=colors, edgecolor="white")
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"T{int(t)} {l}" for t, l in
                    zip(d["dominant_topic"], d["topic_label"])], fontsize=9)
ax.axvline(0, color=GRID, lw=1)
ax.set_xlabel("平均婚姻态度 Attitude toward marriage (mean)", color=INK)
ax.set_title("主题 × 态度：哪些议题把婚姻讲得更正面？\n"
             "Topic × attitude: which topics frame marriage more positively?",
             color=INK, fontsize=13)
ax.grid(axis="x", ls=":", color=GRID)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_attitude_by_topic.png"), dpi=200)
plt.close()

# (d) attitude by media outlet
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(by_media)), by_media["mean_attitude"], color=C_POS,
       edgecolor="white")
ax.set_xticks(range(len(by_media)))
ax.set_xticklabels(by_media.index, rotation=30, ha="right", fontsize=9)
ax.axhline(0, color=GRID, lw=1)
ax.set_ylabel("平均婚姻态度 Attitude toward marriage (mean)", color=INK)
ax.set_title("不同官方媒体的婚姻态度 Attitude by official media outlet\n"
             "(outlets with ≥50 highly relevant articles)", color=INK, fontsize=12)
ax.grid(axis="y", ls=":", color=GRID)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_attitude_by_media.png"), dpi=200)
plt.close()

print("=" * 60)
print("All outputs written to:", OUT)

