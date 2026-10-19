# Official-Media Framing and User Discussions of Marriage in China

**SOSC 314 · Ruizhe (Rachel) Wang, Jingwen Zhai, and Congxi Guan**

**GitHub Pages report:** Link to be added.

**[Read the report in Colab](https://colab.research.google.com/drive/1FiEUvvVwBbwuYdd1gTovuYx2q4KIjuTR)**

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

## Reproduce the Analysis

The repository groups data in `data/`, saved figures in `figure/`, and code in:

```text
scripts/
├── web_scraping/
├── data_cleaning/
├── topic_modeling/       # includes Fightin’ Words
├── attitude_analysis/
├── diagnostics/
└── visualization/
```

### Environment

Use Python 3.11 in a virtual environment, or Google Colab. A GPU is useful for Transformer training.

```bash
git clone https://github.com/wrz-r/SOSC314_Rachel-Wang_Jingwen-Zhai_Congxi-Guan.git
cd SOSC314_Rachel-Wang_Jingwen-Zhai_Congxi-Guan
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### Prepare the Inputs

- Start with the included preprocessed data to reproduce the vocabulary and topic analyses. Extract `weibo_posts_analysis.zip` first.
- The notebooks retain their original layout assumptions: create an empty `notebook/` directory in the repository root, and copy both analysis CSVs into `data/`. Open notebooks from the repository root or their own directories.
- To repeat preprocessing, also copy the two cleaned CSVs into `data/` and supply the four stopword files named in `preprocess_text.ipynb` under `data/stopwords/`.
- For shared-issue keyword comparisons, supply `data/analysis_results/lda_k12_presentation/k12_keyword_weights.csv`. For attitude analysis, set the script paths and provide its labelled data, model files, and any project modules it imports, including `attitude_lexicon.py` for the dictionary pipeline.
- To recollect data, use `scripts/web_scraping/official website.py` and `final_weibo_datacollect.py`. Set the intended dates and sampling settings first; the Weibo workflow uses Colab and requires your own cookie. `user_initiated_posts.py` and `settings.py` retain the earlier pilot workflow.

### Run Order

Run notebook cells from top to bottom:

1. **Preprocessing, if needed:** `scripts/data_cleaning/preprocess_text.ipynb`.
2. **Vocabulary comparison:** `scripts/topic_modeling/fightin_words_comparison.ipynb`.
3. **Topic models:** `lda_official_media.ipynb`, then `lda_weibo_posts.ipynb`, both in `scripts/topic_modeling/`.
4. **Topic summaries:** `scripts/topic_modeling/lda_k12_topic_summary.ipynb`.
5. **Topic diagnostics:** `lda_topic_quality.ipynb` and `lda_topic_count_sensitivity.ipynb` in `scripts/diagnostics/`.
6. **Attitudes and robustness:** Configure inputs in `scripts/attitude_analysis/`, run the chosen attitude model, then the matching checks in `scripts/diagnostics/`. Use `scripts/visualization/` for the corresponding plotting code; some plotting files require variables from the preceding analysis session.

Notebook tables are saved under `data/analysis_results/`; generated plots are saved under `data/generated_figures/`. Earlier project figures remain in `figure/`.
