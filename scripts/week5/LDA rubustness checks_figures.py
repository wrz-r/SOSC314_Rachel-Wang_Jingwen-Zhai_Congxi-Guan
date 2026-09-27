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



