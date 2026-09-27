# Visualization: specification curve and substantive-result stability
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 300,
})

COLORS = {
    "anchor_jackknife": "#315A7D",
    "min_frequency": "#E07A5F",
    "aggregation": "#75608A",
    "baseline": "#3A9D8F",
    "grid": "#D9DEE5",
}

anchor = summary[summary["group"] == "anchor_jackknife"].copy()
anchor = anchor.sort_values("spearman_vs_baseline").reset_index(drop=True)
other = summary[summary["group"].isin(["min_frequency", "aggregation"])].copy()
other = other.sort_values(["group", "spec_id"]).reset_index(drop=True)
all_alt = summary[summary["group"] != "baseline"].copy().reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: all 25 balanced anchor omissions.
axes[0].axhspan(0.90, 1.005, color="#DCEFE1", alpha=0.8)
axes[0].axhspan(0.80, 0.90, color="#F8E8AF", alpha=0.7)
axes[0].scatter(np.arange(1, len(anchor) + 1), anchor["spearman_vs_baseline"],
                color=COLORS["anchor_jackknife"], s=40)
axes[0].plot(np.arange(1, len(anchor) + 1), anchor["spearman_vs_baseline"],
             color=COLORS["anchor_jackknife"], lw=1.2, alpha=0.7)
axes[0].axhline(anchor["spearman_vs_baseline"].median(), color="#333333",
                ls="--", lw=1.2,
                label=f"Median = {anchor['spearman_vs_baseline'].median():.3f}")
axes[0].set_ylim(max(0.0, anchor["spearman_vs_baseline"].min() - 0.05), 1.01)
axes[0].set_xlabel("Anchor-pair omission, ordered by stability")
axes[0].set_ylabel("Spearman correlation with baseline scores")
axes[0].set_title("A. Balanced anchor-pair jackknife")
axes[0].grid(axis="y", color=COLORS["grid"], lw=0.7)
axes[0].legend(frameon=False, loc="lower right")

# Panel B: vocabulary and aggregation alternatives.
labels = [
    (f"MIN_FREQ={int(row.min_freq)}" if row.group == "min_frequency"
     else row.aggregation.replace("_", " ").title())
    for row in other.itertuples()
]
y = np.arange(len(other))
bar_colors = [COLORS[group] for group in other["group"]]
axes[1].barh(y, other["spearman_vs_baseline"], color=bar_colors, alpha=0.9)
axes[1].axvline(0.90, color="#777777", ls="--", lw=1)
axes[1].set_yticks(y)
axes[1].set_yticklabels(labels)
axes[1].set_xlim(min(0.75, other["spearman_vs_baseline"].min() - 0.03), 1.01)
axes[1].set_xlabel("Spearman correlation with baseline scores")
axes[1].set_title("B. Alternative internal settings")
axes[1].grid(axis="x", color=COLORS["grid"], lw=0.7)
for yi, value in zip(y, other["spearman_vs_baseline"]):
    axes[1].text(value - 0.005, yi, f"{value:.3f}", ha="right", va="center",
                 color="white" if value < 0.985 else "#333333", fontsize=9)

# Panel C: do the corpus-level attitude conclusions change?
x = np.arange(len(all_alt))
axes[2].scatter(x, all_alt["official_mean"], s=42, color="#E07A5F",
                label="Official-media mean")
axes[2].scatter(x, all_alt["weibo_mean"], s=42, color="#3A9D8F",
                marker="D", label="Weibo mean")
axes[2].axhline(baseline_official_mean, color="#E07A5F", lw=1.5, ls=":",
                label=f"Official baseline = {baseline_official_mean:.3f}")
axes[2].axhline(baseline_weibo_mean, color="#3A9D8F", lw=1.5, ls=":",
                label=f"Weibo baseline = {baseline_weibo_mean:.3f}")
axes[2].axhline(0, color="#555555", ls="--", lw=1)
axes[2].set_xlabel("Alternative specification")
axes[2].set_ylabel("Mean Semantic Scaling score")
axes[2].set_title("C. Stability of corpus-level conclusions")
axes[2].grid(axis="y", color=COLORS["grid"], lw=0.7)
axes[2].legend(frameon=False, fontsize=8, loc="best")

fig.suptitle("Semantic Scaling robustness across internal settings",
             fontsize=17, fontweight="bold", y=1.01)
fig.text(
    0.5, -0.02,
    "Note: The embedding model is fixed. Green shading denotes score correlation "
    ">= .90; yellow denotes .80-.90. Panel C tests whether the estimated "
    "corpus-level mean attitudes retain their direction and similar magnitude.",
    ha="center", va="top", fontsize=9, color="#555555", wrap=True,
)
fig.tight_layout(w_pad=2.5)

figure_png = OUTPUT_DIR / "semantic_scaling_robustness_figure.png"
figure_pdf = OUTPUT_DIR / "semantic_scaling_robustness_figure.pdf"
fig.savefig(figure_png, bbox_inches="tight", facecolor="white")
fig.savefig(figure_pdf, bbox_inches="tight", facecolor="white")
plt.show()



# Transparent PASS/PARTIAL/FAIL assessment table
def assess_group(frame):
    minimum_rho = float(frame["spearman_vs_baseline"].min())
    median_rho = float(frame["spearman_vs_baseline"].median())
    official_sign_stability = float(frame["official_sign_same_as_baseline"].astype(float).mean())
    gap_stability = float(frame["gap_sign_same_as_baseline"].astype(float).mean())
    profile_minimum = float(frame["year_source_profile_spearman"].min())
    if minimum_rho >= 0.90 and official_sign_stability == 1.0 and profile_minimum >= 0.90:
        status = "PASS"
    elif median_rho >= 0.80 and official_sign_stability >= 0.80:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return (minimum_rho, median_rho, profile_minimum,
            official_sign_stability, gap_stability, status)


assessment_rows = []
for group, label in [
    ("anchor_jackknife", "Anchor-pair jackknife"),
    ("min_frequency", "Vocabulary MIN_FREQ"),
    ("aggregation", "Document aggregation"),
]:
    frame = summary[summary["group"] == group]
    (minimum_rho, median_rho, profile_minimum,
     official_sign_stability, gap_stability, status) = assess_group(frame)
    assessment_rows.append({
        "check": label,
        "minimum_score_correlation": minimum_rho,
        "median_score_correlation": median_rho,
        "minimum_year_profile_correlation": profile_minimum,
        "official_sign_stability": official_sign_stability,
        "gap_sign_stability": gap_stability,
        "assessment": status,
        "evidence": (
            f"min rho={minimum_rho:.2f}; median rho={median_rho:.2f}; "
            f"min year-profile rho={profile_minimum:.2f}; "
            f"official sign stable={official_sign_stability:.0%}; "
            f"gap sign stable={gap_stability:.0%}"
        ),
    })

assessment = pd.DataFrame(assessment_rows)
assessment.to_csv(OUTPUT_DIR / "semantic_scaling_robustness_assessment.csv",
                  index=False, encoding="utf-8-sig")

# More reliable table: show each robustness metric in a separate column.
STATUS_COLOR = {
    "PASS": "#CFE8D5",
    "PARTIAL": "#F7E3A1",
    "FAIL": "#F3C2BE"
}

# Prepare short, single-line cells.
cell_text = []

for _, row in assessment.iterrows():
    cell_text.append([
        row["check"],
        f"{row['minimum_score_correlation']:.2f}",
        f"{row['median_score_correlation']:.2f}",
        f"{row['minimum_year_profile_correlation']:.2f}",
        f"{row['official_sign_stability']:.0%}",
        f"{row['gap_sign_stability']:.0%}",
        row["assessment"]
    ])

fig, ax = plt.subplots(figsize=(18, 5.2))
ax.axis("off")

table = ax.table(
    cellText=cell_text,
    colLabels=[
        "Internal-setting check",
        "Minimum\nscore rho",
        "Median\nscore rho",
        "Minimum year-\nprofile rho",
        "Official-sign\nstability",
        "Gap-sign\nstability",
        "Assessment"
    ],
    colWidths=[
        0.23,
        0.12,
        0.12,
        0.15,
        0.13,
        0.12,
        0.13
    ],
    cellLoc="center",
    bbox=[0.01, 0.29, 0.98, 0.48]
)

table.auto_set_font_size(False)
table.set_fontsize(10)

# Format header cells.
for column in range(7):
    header_cell = table[(0, column)]
    header_cell.set_facecolor("#315A7D")
    header_cell.set_text_props(
        color="white",
        weight="bold",
        ha="center",
        va="center",
        fontsize=10
    )
    header_cell.set_height(0.16)

# Format data rows.
for row_number, status in enumerate(
    assessment["assessment"],
    start=1
):
    background = (
        "#F7F8FA"
        if row_number % 2
        else "white"
    )

    for column in range(7):
        cell = table[(row_number, column)]
        cell.set_height(0.13)
        cell.set_facecolor(background)
        cell.set_text_props(
            ha="center",
            va="center",
            fontsize=10
        )

    # Left-align the check name.
    table[(row_number, 0)].set_text_props(
        ha="left",
        va="center",
        fontsize=10
    )

    # Color the final assessment cell.
    table[(row_number, 6)].set_facecolor(
        STATUS_COLOR[status]
    )
    table[(row_number, 6)].set_text_props(
        weight="bold",
        ha="center",
        va="center",
        fontsize=10.5
    )

ax.set_title(
    "Semantic Scaling Robustness Assessment",
    fontsize=17,
    fontweight="bold",
    pad=20
)

fig.text(
    0.5,
    0.105,
    ("Note: PASS requires minimum document-score and year-profile correlations ≥ .90 and 100% stability in the direction of the official-media mean. PARTIAL requires median score correlation ≥ .80 and official-mean direction stability ≥ 80%. "
    "The Official–Weibo gap is treated as a secondary result."),
    ha="center",
    va="center",
    fontsize=9.5,
    color="#555555",
    wrap=True
)

fig.text(
    0.5,
    0.055,
    ("These classifications are transparent descriptive thresholds not statistical significance tests."),
    ha="center",
    va="center",
    fontsize=9,
    color="#777777"
)

table_png = (
    OUTPUT_DIR
    / "semantic_scaling_robustness_assessment.png"
)

table_pdf = (
    OUTPUT_DIR
    / "semantic_scaling_robustness_assessment.pdf"
)

fig.savefig(
    table_png,
    bbox_inches="tight",
    facecolor="white",
    dpi=300
)

fig.savefig(
    table_pdf,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nRobustness assessment")
print(assessment[["check", "evidence", "assessment"]].to_string(index=False))


