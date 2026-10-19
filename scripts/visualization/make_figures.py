import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



# 1. Load data
file_path = Path(
    "/Users/congxi/weibo-search/weibo_2021_marriage_attitude_coded.csv"
)

df = pd.read_csv(
    file_path,
    encoding="utf-8-sig"
)

print("Data loaded successfully.")
print(f"Number of posts: {len(df)}")



# 2. Define keywords
# All keywords for Figure 1
keywords = [
    "结婚",
    "婚姻",
    "恋爱",
    "对象",
    "伴侣",
    "相亲",
    "单身",
    "不婚",
    "晚婚",
    "恐婚",
    "生育",
    "生孩子",
    "生娃",
    "出生率",
    "生育率",
    "二孩",
    "三孩",
    "二胎",
    "三胎"
]


# Marriage-related keywords only
# Used for Figure 2 because "marriage_attitude"
# measures attitudes toward marriage, not fertility.
marriage_keywords = [
    "结婚",
    "婚姻",
    "恋爱",
    "对象",
    "伴侣",
    "相亲",
    "单身",
    "不婚",
    "晚婚",
    "恐婚"
]



# 3. English labels
keyword_labels = {
    "结婚": "Getting married",
    "婚姻": "Marriage",
    "恋爱": "Romantic relationship",
    "对象": "Partner",
    "伴侣": "Partner",
    "相亲": "Matchmaking",
    "单身": "Singlehood",
    "不婚": "Non-marriage",
    "晚婚": "Late marriage",
    "恐婚": "Fear of marriage",
    "生育": "Fertility",
    "生孩子": "Having children",
    "生娃": "Having children",
    "出生率": "Birth rate",
    "生育率": "Fertility rate",
    "二孩": "Second child",
    "三孩": "Third child",
    "二胎": "Second child",
    "三胎": "Third child"
}


# 4. Create keyword-level dataset

keyword_rows = []

for _, row in df.iterrows():

    matched = str(row["matched_keywords"])

    matched_list = [
        x.strip()
        for x in matched.split(";")
        if x.strip()
    ]

    for keyword in matched_list:

        if keyword in keywords:

            keyword_rows.append({
                "post_id": row["post_id"],
                "keyword": keyword,
                "marriage_attitude": row["marriage_attitude"]
            })


keyword_df = pd.DataFrame(keyword_rows)

print(
    f"Keyword-level observations: {len(keyword_df)}"
)


# Figure 1
# Frequency of Marriage and Fertility Keywords

keyword_frequency = (
    keyword_df
    .groupby("keyword")["post_id"]
    .nunique()
    .reindex(keywords)
    .fillna(0)
)

# Remove keywords that do not appear in the dataset
keyword_frequency = keyword_frequency[
    keyword_frequency > 0
]

# Sort from low to high
# Then invert y-axis so the highest appears at the top
keyword_frequency = keyword_frequency.sort_values(
    ascending=True
)

# English labels
english_labels = [
    keyword_labels[k]
    for k in keyword_frequency.index
]



# Create Figure 1


fig, ax = plt.subplots(
    figsize=(10, 7)
)

# Muted blue
fig1_color = "#6F9FB5"

ax.barh(
    english_labels,
    keyword_frequency.values,
    color=fig1_color
)

# Highest frequency at top
ax.invert_yaxis()

# Labels
ax.set_xlabel(
    "Number of posts",
    fontsize=12
)

ax.set_ylabel(
    "Keyword",
    fontsize=12
)

ax.set_title(
    "Figure 1. Frequency of Marriage and Fertility Keywords",
    fontsize=15,
    pad=12
)

# Remove unnecessary top/right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light grid on x-axis
ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)

# Put grid behind bars
ax.set_axisbelow(True)

plt.tight_layout()


# Save Figure 1
fig1_path = (
    file_path.parent /
    "figure1_keyword_frequency.png"
)

plt.savefig(
    fig1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Figure 1 saved to: {fig1_path}"
)



# Figure 2
# Marriage Keyword × Marriage Attitude


# Keep ONLY marriage-related keywords
marriage_df = keyword_df[
    keyword_df["keyword"].isin(marriage_keywords)
].copy()



# Count posts by keyword and attitude
attitude_counts = (
    marriage_df
    .groupby(
        ["keyword", "marriage_attitude"]
    )["post_id"]
    .nunique()
    .unstack(
        fill_value=0
    )
)


# Make sure all three attitude categories exist
for attitude in [-1, 0, 1]:

    if attitude not in attitude_counts.columns:

        attitude_counts[attitude] = 0


# Order:
# Negative → Neutral → Positive

attitude_counts = attitude_counts[
    [-1, 0, 1]
]



# Convert counts to percentages
attitude_pct = (
    attitude_counts
    .div(
        attitude_counts.sum(axis=1),
        axis=0
    )
    * 100
)


# Keep keyword order
attitude_pct = attitude_pct.reindex(
    marriage_keywords
)

# Remove keywords with no observations
attitude_pct = attitude_pct.dropna(
    how="all"
)

# English labels
english_marriage_labels = [
    keyword_labels[k]
    for k in attitude_pct.index
]



# Figure 2 — Keyword × Attitude

attitude_counts = (
    keyword_df
    .groupby(["keyword", "marriage_attitude"])["post_id"]
    .nunique()
    .unstack(fill_value=0)
)

for attitude in [-1, 0, 1]:
    if attitude not in attitude_counts.columns:
        attitude_counts[attitude] = 0

attitude_counts = attitude_counts[[-1, 0, 1]]

attitude_pct = attitude_counts.div(
    attitude_counts.sum(axis=1),
    axis=0
) * 100

attitude_pct = attitude_pct.reindex(keywords)

# English labels
english_labels = [
    keyword_labels[k] for k in attitude_pct.index
]

fig, ax = plt.subplots(figsize=(12, 7))

bottom = pd.Series(0, index=attitude_pct.index)

labels = {
    -1: "Negative",
    0: "Neutral / Unclear",
    1: "Positive"
}

for attitude in [-1, 0, 1]:

    values = attitude_pct[attitude]

    ax.bar(
        english_labels,
        values,
        bottom=bottom,
        label=labels[attitude]
    )

    bottom += values

ax.set_ylabel("Percentage of posts (%)")
ax.set_xlabel("Keyword")

ax.set_title(
    "Figure 2. Marriage/Fertility Attitude by Keyword"
)

ax.set_ylim(0, 100)

ax.legend(
    title="Attitude",
    loc="upper right"
)

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

fig2_path = file_path.parent / "figure2_keyword_attitude.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Figure 2 saved to: {fig2_path}")


# Formatting

ax.set_ylabel(
    "Percentage of posts (%)",
    fontsize=12
)

ax.set_xlabel(
    "Marriage-related keyword",
    fontsize=12
)

ax.set_title(
    "Figure 2. Marriage Attitude by Keyword",
    fontsize=15,
    pad=12
)

ax.set_ylim(
    0,
    100
)

# X-axis labels
plt.xticks(
    rotation=45,
    ha="right"
)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light horizontal grid
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)


# Legend
ax.legend(
    title="Attitude",
    frameon=False,
    loc="upper right"
)


plt.tight_layout()


# Save Figure 2
fig2_path = (
    file_path.parent /
    "figure2_marriage_keyword_attitude.png"
)

plt.savefig(
    fig2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Figure 2 saved to: {fig2_path}"
)


