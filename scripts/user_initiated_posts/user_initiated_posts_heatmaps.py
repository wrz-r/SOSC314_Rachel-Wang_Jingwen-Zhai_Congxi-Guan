import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the original Chinese search keywords in the order in which they should appear in the heatmap.
KEYWORDS = ["婚姻","结婚","晚婚","不婚","生育","生孩子","不婚不育"]

# Define English translations used only as display labels in the figure.
keyword_labels = {
    "婚姻": "Marriage",
    "结婚": "Getting married",
    "晚婚": "Late marriage",
    "不婚": "Not marrying",
    "生育": "Childbearing",
    "生孩子": "Having children",
    "不婚不育": "No marriage or children"
}
# Define the chronological order of the 12 months in 2025.
month_order = [
    f"2025-{month:02d}"
    for month in range(1, 13)
]

# Count the number of cleaned rows for each search-query and month combination.
cleaned_heatmap_data = (
    df.groupby(
        ["search_query", "month"]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(
        index=KEYWORDS,
        columns=month_order,
        fill_value=0
    )
)

# Replace the Chinese row labels with their English translations for presentation purposes only.
cleaned_heatmap_data = (
    cleaned_heatmap_data.rename(
        index=keyword_labels
    )
)
# Display the completed keyword-by-month count table before using it to create the heatmap.
display(cleaned_heatmap_data)

# Create a figure 
plt.figure(figsize=(14, 6))
# Create a heatmap showing the number of cleaned records retained for each search-query and month combination.
sns.heatmap(
    cleaned_heatmap_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    vmin=0,
    vmax=30,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={
        "label": "Cleaned records retained"
    }
)

plt.title(
    "Availability of Cleaned Marriage- and Childbearing-Related Weibo Posts in 2025",
    fontsize=14,
    pad=15
)

plt.xlabel("Month")
plt.ylabel("Search keyword")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.figtext(
    0.5,
    -0.04,
    ("Note: Each keyword-month query was initially capped at 30 records. Counts show the number retained after excluding irrelevant content and therefore indicate sample availability rather than total Weibo activity."),
    ha="center",
    fontsize=9,
    wrap=True
)

plt.tight_layout()

# Save the completed heatmap as PNG image.
plt.savefig(
    "/content/weibo_2025_cleaned_count_heatmap_english.png",
    dpi=300,
    bbox_inches="tight"
)
# Display the completed heatmap
plt.show()



