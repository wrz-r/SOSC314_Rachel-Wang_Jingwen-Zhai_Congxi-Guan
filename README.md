# Official-Media Framing and User Discussions of Marriage in China

**SOSC 314 · Ruizhe (Rachel) Wang, Jingwen Zhai, and Congxi Guan**

**GitHub Pages report:** GitHub Pages report:** [View the final report](https://jingwenzhai121.github.io/sosc314-project-website/)

## Research Question

How do Chinese official media frame marriage, and how do their topics and attitudes differ from user-initiated Weibo discussions during **2021–2025**?

## Why It Matters

Marriage connects personal decisions with demographic policy, financial pressure, gender expectations, and family responsibilities. Comparing institutional coverage with discussions that users initiate themselves helps identify which concerns receive public attention and where their priorities differ.

## Data

- **Official media:** Full-text articles from People.cn, CCTV, and China News Service, identified through marriage-related keyword searches.
- **Weibo:** Original public posts collected with [weibo-search](https://github.com/dataabc/weibo-search), using marriage-related keywords and time-based search windows.
- **Unit of analysis:** One article or one post. Cleaning removes irrelevant content and duplicates; Chinese text is segmented with `jieba` and filtered with shared stopwords.

We collected the data through **web scraping**. Both the scraping code and the collected, cleaned datasets are provided in this repository.

- **Use the provided data:** Download or clone the repository. Cleaned datasets are in `data/cleaned data/`, and model-ready datasets are in `data/preprocessed data for analysis/`. Extract the Weibo ZIP file before use.
- **Collect the data again:** Use the scripts in `scripts/web_scraping/` for official-media articles and Weibo posts, followed by the cleaning code in `scripts/data_cleaning/`.

The model-ready datasets contain **2,020 official-media articles and 21,945 Weibo posts**.

## Methods

- **Fightin’ Words:** Identify vocabulary that distinguishes official media from Weibo.
- **LDA topic modeling:** Fit separate topic models to each corpus; use **K = 12** for the main comparison and examine topic prevalence and annual trends.
- **Attitude analysis:** Compare semantic scaling with a Transformer classifier for negative, neutral, and positive attitudes toward marriage.
- **Diagnostics:** Assess topic coherence, exclusivity, sensitivity to K, random-seed stability, preprocessing choices, and human–model agreement.

## Main Findings

- Official media emphasize registration, legal procedures, and wedding reform. Weibo users focus more on relationships, family responsibilities, marriage pressure, and personal choices.
- Shared issues such as bride price and divorce receive different emphases: institutional solutions in official media and personal experiences on Weibo.
- The reported attitude results suggest more positive marriage discourse in official media. Small labelled samples, missed negative expressions, and variation across model settings limit the precision of this comparison.

These findings describe the collected texts; they do not establish causal media effects or represent the attitudes of the entire population.

## Repository Structure

```text
.
├── data/                      # Cleaned and preprocessed datasets
├── figure/                    # Saved project figures
├── scripts/
│   ├── web_scraping/           # Official-media and Weibo collection
│   ├── data_cleaning/          # Cleaning and text preprocessing
│   ├── topic_modeling/         # Fightin’ Words and LDA notebooks
│   ├── attitude_analysis/      # Classifier training and semantic scaling
│   ├── diagnostics/            # Model diagnostics and robustness checks
│   └── visualization/          # Plotting code
└── requirements.txt
```

## Reproducibility

### Prerequisites

- Python 3.11 and Git.
- Google Colab can be used for the Colab-based workflows.
- A GPU is recommended for Transformer training.

### 1. Clone and Install

```bash
git clone https://github.com/wrz-r/SOSC314_Rachel-Wang_Jingwen-Zhai_Congxi-Guan.git
cd SOSC314_Rachel-Wang_Jingwen-Zhai_Congxi-Guan
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare the Data

The collected and cleaned datasets are included. To start with the model-ready data, run the following from the repository root. This places the inputs where the existing notebooks expect them and creates the directory used by their project-root check.

```python
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

root = Path.cwd()
(root / "notebook").mkdir(exist_ok=True)
data = root / "data"
inputs = data / "preprocessed data for analysis"
copy2(inputs / "official_media_analysis.csv", data / "official_media_analysis.csv")
with ZipFile(inputs / "weibo_posts_analysis.zip") as archive:
    with archive.open("weibo_posts_analysis.csv") as source:
        (data / "weibo_posts_analysis.csv").write_bytes(source.read())
```

To repeat earlier stages:

- **Collection:** Use `scripts/web_scraping/official website.py` and `final_weibo_datacollect.py`. Configure dates and sampling settings first; the Weibo workflow runs in Colab with your own cookie. The older `user_initiated_posts.py` and `settings.py` provide the pilot workflow.
- **Preprocessing:** Copy the cleaned CSVs from `data/cleaned data/` into `data/`, place the four stopword lists named in `preprocess_text.ipynb` under `data/stopwords/`, then run that notebook.

### 3. Run the Analysis

Launch Jupyter from the repository root:

```bash
python -m notebook
```

Run each notebook from top to bottom, in this order:

1. **Vocabulary:** `scripts/topic_modeling/fightin_words_comparison.ipynb`.
2. **Topics:** `lda_official_media.ipynb`, followed by `lda_weibo_posts.ipynb`, in `scripts/topic_modeling/`.
3. **Summary:** `scripts/topic_modeling/lda_k12_topic_summary.ipynb`.
4. **Diagnostics:** `lda_topic_quality.ipynb` and `lda_topic_count_sensitivity.ipynb` in `scripts/diagnostics/`.

For shared-issue keyword comparisons, run `lda_shared_issue_keywords.ipynb` after placing `k12_keyword_weights.csv` in `data/analysis_results/lda_k12_presentation/`.

For attitude analysis, configure the paths and required labelled data/model files in `scripts/attitude_analysis/`, then run the corresponding checks in `scripts/diagnostics/`. The older dictionary pipeline imports the project module `attitude_lexicon.py`. Plotting code is in `scripts/visualization/`; session-dependent plots should follow their matching analysis in the same session.

### Outputs

- **Result tables:** `data/analysis_results/`.
- **Notebook figures:** `data/generated_figures/`.
- **Previously saved figures:** `figure/`.
