# -*- coding: utf-8 -*-
"""Make one coherence-exclusivity plot for each LDA corpus."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parent
SOURCES = [
    ("Official media", BASE / "lda_qualitative_k_results/topic_diagnostics.csv",
     "#287D8E", BASE / "official_coherence_exclusivity.png"),
    ("Weibo posts", BASE / "weibo_lda_results/topic_diagnostics.csv",
     "#D27A61", BASE / "weibo_coherence_exclusivity.png"),
]
LABEL_OFFSETS = {
    "Official media": {1: (-12, -12), 3: (2, 8), 4: (7, -9),
                       7: (5, -10), 8: (-12, -10), 11: (7, 5), 12: (7, 5)},
    "Weibo posts": {4: (8, 8), 7: (-14, 7), 10: (-20, -11), 15: (7, -10)},
}


def main():
    datasets = [(title, pd.read_csv(path), color, output)
                for title, path, color, output in SOURCES]

    # Both figures use the same axis limits.
    x_values = pd.concat([data["coherence_c_v"] for _, data, _, _ in datasets])
    y_values = pd.concat([data["exclusivity_mean"] for _, data, _, _ in datasets])
    x_limits = (max(0, x_values.min() - 0.06), min(1, x_values.max() + 0.06))
    y_limits = (max(0, y_values.min() - 0.08), min(1, y_values.max() + 0.08))

    for title, data, color, output in datasets:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(data["coherence_c_v"], data["exclusivity_mean"],
                   s=85, color=color, alpha=0.85)

        for _, row in data.iterrows():
            topic = int(row["topic"])
            offset = LABEL_OFFSETS[title].get(topic, (5, 5))
            ax.annotate(str(topic),
                        (row["coherence_c_v"], row["exclusivity_mean"]),
                        xytext=offset, textcoords="offset points", fontsize=9)

        ax.set(title=f"{title}: topic diagnostics (K={len(data)})",
               xlabel="Coherence (c_v)",
               ylabel="Exclusivity (mean word share)",
               xlim=x_limits, ylim=y_limits)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
