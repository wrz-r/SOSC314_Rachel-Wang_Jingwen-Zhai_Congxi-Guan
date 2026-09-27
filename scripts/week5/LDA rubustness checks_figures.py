import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {
    "coherence": "#315A7D",
    "diversity": "#E07A5F",
    "alignment": "#3A9D8F",
    "jaccard": "#75608A",
    "minimum": "#A7A9AC",
    "baseline": "#444444",
    "grid": "#D9DEE5",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 300,
})


def plot_quality_panel(ax, corpus_name):
    folder = OUTPUT_ROOT / corpus_name
    cfg = CORPUS_CONFIGS[corpus_name]
    metrics = pd.read_csv(folder / "model_metrics.csv")
    matches = pd.read_csv(folder / "check1_k_topic_matches.csv")

    quality = (
        metrics.loc[metrics["seed"] == REFERENCE_SEED,
                    ["k", "mean_npmi_coherence", "topic_diversity"]]
        .drop_duplicates("k")
        .sort_values("k")
    )
    alignment = (
        matches.groupby("candidate_k", as_index=False)
        .agg(mean_topic_alignment=("cosine_similarity", "mean"))
        .rename(columns={"candidate_k": "k"})
    )
    quality = quality.merge(alignment, on="k", how="left")
    quality.to_csv(
        folder / "figure_data_k_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ax.plot(quality["k"], quality["mean_npmi_coherence"], marker="o", lw=2.2,
            color=COLORS["coherence"], label="Mean NPMI coherence")
    ax.plot(quality["k"], quality["topic_diversity"], marker="s", lw=2.2,
            color=COLORS["diversity"], label="Topic diversity")
    ax.plot(quality["k"], quality["mean_topic_alignment"], marker="D", lw=2.2,
            color=COLORS["alignment"], label="Alignment with baseline")
    ax.axvline(cfg["baseline_k"], color=COLORS["baseline"], lw=1.2,
               ls="--", alpha=0.85)
    ax.text(cfg["baseline_k"], 1.025, f" Baseline K={cfg['baseline_k']}",
            ha="center", va="bottom", fontsize=9, color=COLORS["baseline"])
    ax.set_xticks(K_VALUES)
    lower = min(-0.05, float(quality["mean_npmi_coherence"].min()) - 0.05)
    ax.set_ylim(lower, 1.08)
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("Quality or similarity score")
    ax.set_title(f"{corpus_name.title()}: sensitivity to topic number")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.8, alpha=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def plot_seed_panel(ax, corpus_name):
    folder = OUTPUT_ROOT / corpus_name
    stability = pd.read_csv(folder / "check2_seed_stability_summary.csv")
    # Seed 42 is the reference model. Its similarity with itself is exactly 1
    # by construction, so exclude it from the visual comparison while keeping
    # it in the raw result files as the necessary matching reference.
    stability = stability[stability["candidate_seed"] != REFERENCE_SEED].copy()
    stability = stability.sort_values("candidate_seed").reset_index(drop=True)
    stability.to_csv(
        folder / "figure_data_seed_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    y = np.arange(len(stability))
    for yi, row in stability.iterrows():
        ax.hlines(
            yi,
            row["minimum_topic_cosine"],
            row["mean_topic_cosine"],
            color=COLORS["minimum"],
            lw=3,
            alpha=0.9,
        )
    ax.scatter(stability["mean_topic_cosine"], y, s=60,
               color=COLORS["alignment"], label="Mean topic cosine", zorder=3)
    ax.scatter(stability["mean_topword_jaccard"], y, s=55, marker="s",
               color=COLORS["jaccard"], label="Mean top-word Jaccard", zorder=3)
    ax.scatter(stability["minimum_topic_cosine"], y, s=35, marker="|",
               color=COLORS["baseline"], label="Minimum topic cosine", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Seed {int(seed)}" for seed in stability["candidate_seed"]])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Stability relative to seed 42")
    ax.set_ylabel("Random initialization")
    ax.set_title(f"{corpus_name.title()}: stability across random seeds")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.8, alpha=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")


corpora = list(INPUT_FILES)
fig, axes = plt.subplots(
    len(corpora), 2,
    figsize=(14, 5.0 * len(corpora)),
    squeeze=False,
)
for row_number, corpus_name in enumerate(corpora):
    plot_quality_panel(axes[row_number, 0], corpus_name)
    plot_seed_panel(axes[row_number, 1], corpus_name)

fig.suptitle(
    "Robustness of LDA topic solutions",
    fontsize=17,
    fontweight="bold",
    y=0.995,
)
fig.text(
    0.5,
    0.005,
    "Note: Topics are aligned to the reported baseline with Hungarian matching "
    "on topic-word cosine similarity. Higher values indicate stronger stability. "
    "In-sample perplexity is retained in the accompanying CSV but is not used as "
    "the sole criterion for selecting K.",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
    wrap=True,
)
fig.tight_layout(rect=[0, 0.045, 1, 0.97], h_pad=3.0, w_pad=2.5)

png_path = OUTPUT_ROOT / "LDA_robustness_summary.png"
pdf_path = OUTPUT_ROOT / "LDA_robustness_summary.pdf"
fig.savefig(png_path, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.show()
print(f"Saved visualization: {png_path}")
print(f"Saved print-quality version: {pdf_path}")


# These cutoffs are transparent project-level interpretation rules, not universal statistical significance thresholds.
def assess_k(mean_alignment, weakest_alignment):
    if mean_alignment >= 0.75 and weakest_alignment >= 0.65:
        return "PASS"
    if mean_alignment >= 0.60 and weakest_alignment >= 0.50:
        return "PARTIAL"
    return "FAIL"


def assess_seed(mean_cosine, mean_jaccard, mean_weakest_cosine):
    if mean_cosine >= 0.75 and mean_jaccard >= 0.50 and mean_weakest_cosine >= 0.40:
        return "PASS"
    if mean_cosine >= 0.60 and mean_jaccard >= 0.25:
        return "PARTIAL"
    return "FAIL"


decision_rows = []
for corpus_name in corpora:
    folder = OUTPUT_ROOT / corpus_name

    k_raw = pd.read_csv(folder / "check1_k_topic_matches.csv")
    k_alt = k_raw[k_raw["candidate_k"] != k_raw["reference_k"]].copy()
    k_by_model = (
        k_alt.groupby("candidate_k", as_index=False)
        .agg(mean_alignment=("cosine_similarity", "mean"),
             mean_jaccard=("topword_jaccard", "mean"))
    )
    k_mean = float(k_by_model["mean_alignment"].mean())
    k_weak = float(k_by_model["mean_alignment"].min())
    k_jac = float(k_by_model["mean_jaccard"].mean())
    decision_rows.append({
        "corpus": corpus_name.title(),
        "check": "Alternative K",
        "mean_similarity": k_mean,
        "weakest_similarity": k_weak,
        "topword_jaccard": k_jac,
        "assessment": assess_k(k_mean, k_weak),
        "evidence": (
            f"mean alignment={k_mean:.2f}; lowest K={k_weak:.2f}; "
            f"top-word overlap={k_jac:.2f}"
        ),
    })

    seed_raw = pd.read_csv(folder / "check2_seed_stability_summary.csv")
    seed_alt = seed_raw[seed_raw["candidate_seed"] != REFERENCE_SEED].copy()
    seed_mean = float(seed_alt["mean_topic_cosine"].mean())
    seed_weak = float(seed_alt["minimum_topic_cosine"].mean())
    seed_jac = float(seed_alt["mean_topword_jaccard"].mean())
    decision_rows.append({
        "corpus": corpus_name.title(),
        "check": "Alternative seeds",
        "mean_similarity": seed_mean,
        "weakest_similarity": seed_weak,
        "topword_jaccard": seed_jac,
        "assessment": assess_seed(seed_mean, seed_jac, seed_weak),
        "evidence": (
            f"mean cosine={seed_mean:.2f}; weakest-topic avg={seed_weak:.2f}; "
            f"top-word overlap={seed_jac:.2f}"
        ),
    })

decisions = pd.DataFrame(decision_rows)
decisions.to_csv(
    OUTPUT_ROOT / "robustness_assessment_table.csv",
    index=False,
    encoding="utf-8-sig",
)

STATUS_COLOR = {"PASS": "#CFE8D5", "PARTIAL": "#F7E3A1", "FAIL": "#F3C2BE"}
cell_text = decisions[["corpus", "check", "evidence", "assessment"]].values.tolist()

fig_height = 2.8 + 0.65 * len(cell_text)
fig, ax = plt.subplots(figsize=(15, fig_height))
ax.axis("off")
table = ax.table(
    cellText=cell_text,
    colLabels=["Corpus", "Robustness check", "Main evidence", "Assessment"],
    colWidths=[0.13, 0.18, 0.52, 0.14],
    cellLoc="left",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.7)

for col in range(4):
    table[(0, col)].set_facecolor("#315A7D")
    table[(0, col)].set_text_props(color="white", weight="bold")
for row_number, status in enumerate(decisions["assessment"], start=1):
    table[(row_number, 3)].set_facecolor(STATUS_COLOR[status])
    table[(row_number, 3)].set_text_props(weight="bold", ha="center")
    for col in range(3):
        table[(row_number, col)].set_facecolor("#F7F8FA" if row_number % 2 else "white")

ax.set_title(
    "LDA robustness assessment",
    fontsize=17,
    fontweight="bold",
    pad=18,
)
fig.text(
    0.5,
    0.03,
    "Decision rules (declared for interpretation): Alternative K passes when "
    "mean alignment >= .75 and the weakest alternative K >= .65. Seed stability "
    "passes when mean cosine >= .75, top-word Jaccard >= .50, and average "
    "weakest-topic cosine >= .40. PARTIAL indicates moderate but non-uniform "
    "stability; these are descriptive thresholds, not significance tests.",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
    wrap=True,
)
fig.tight_layout(rect=[0.01, 0.10, 0.99, 0.93])

assessment_png = OUTPUT_ROOT / "LDA_robustness_assessment_table.png"
assessment_pdf = OUTPUT_ROOT / "LDA_robustness_assessment_table.pdf"
fig.savefig(assessment_png, bbox_inches="tight", facecolor="white")
fig.savefig(assessment_pdf, bbox_inches="tight", facecolor="white")
plt.show()
print("\nRobustness assessment")
print(decisions[["corpus", "check", "evidence", "assessment"]].to_string(index=False))
print(f"Saved assessment table: {assessment_png}")



