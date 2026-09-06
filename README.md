# SOSC314_Rachel-Wang_Jingwen-Zhai_Congxi-Guan
# Sentiment Toward Marriage and Fertility on Weibo

**Group members:** Ruizhe (Rachel) Wang, Jingwen Zhai, and Congxi Guan.

## Research Question

**How does sentiment toward marriage and fertility differ between user comments on official-media Weibo posts and user-initiated Weibo posts, and how has this difference changed from 2021 to 2025?**

## Background

Previous studies have used Weibo to examine public attitudes toward marriage and fertility in China. These discussions often involve economic pressure, parenting costs, gender roles, personal autonomy, and family responsibilities.

However, it remains unclear whether users express different sentiments when responding to official-media content compared with when they independently initiate discussions. This project addresses this gap by comparing the two types of Weibo discourse.

## Data Source

The primary data source is **Weibo (微博)**. The project uses the open-source tool [dataabc/weibo-search](https://github.com/dataabc/weibo-search), together with project-specific Python scripts for screening posts and collecting public comments.

The dataset contains only publicly available Weibo content and consists of two corpora:

1. **Official-media comment corpus:** public user comments under relevant posts published by selected verified official-media accounts, including People’s Daily, CCTV News, Xinhua News Agency, China News Service, China Daily, and Healthy China.

2. **User-initiated post corpus:** original public Weibo posts independently written by users about marriage, fertility, childrearing, or related attitudes.

The unit of analysis is one individual user comment for the official-media corpus and one original user post for the user-initiated corpus.

## Time Period and Keywords

The study covers content posted from **1 January 2021 to 31 December 2025**. The repository currently contains 2025 pilot data.

Keywords include:

- **Marriage and relationships:** `结婚`, `婚姻`, `恋爱`, `相亲`, `单身`, `不婚`, `晚婚`, `恐婚`
- **Fertility and childrearing:** `生育`, `生孩子`, `生娃`, `出生率`, `生育率`, `二孩`, `三孩`, `育儿`, `养娃`
- **Attitudes and constraints:** `生育意愿`, `不想生`, `不敢生`, `生不起`, `养不起`, `催婚`

## Inclusion and Exclusion Criteria

Content is included when it:

- is publicly available on Weibo;
- was posted between 2021 and 2025;
- contains relevant marriage, fertility, or family-related keywords;
- is either a comment under a selected official-media post or an original user-initiated post.

The project excludes duplicate content, advertisements, spam, obvious bot-generated material, unusable text, and posts that contain relevant keywords but are unrelated to marriage, fertility, or family formation.

## Method

1. Search Weibo by keyword and time period.
2. Screen official-media posts using a predefined account list and relevance criteria.
3. Collect public comments under selected official-media posts.
4. Collect relevant original user-initiated posts.
5. Compare sentiment between the two corpora and examine changes over time.

## Repository Structure

```text
.
├── README.md
├── literature review.md
│
├── data/
│   ├── official_media_comments_2025.csv
│   └── weibo_marriage_fertility_2025_final.csv
│
├── scripts/
│   ├── user_initiated_posts.py
│   ├── user_initiated_posts_heatmaps.py
│   └── official_media_comment_scraper/
│       ├── run_weibo_search_pilot.py
│       ├── prepare_pilot_sample.py
│       ├── prepare_official_media_posts.py
│       ├── collect_sample_comments.py
│       └── export_official_cookie_comments.py
│
└── figures/
    ├── week2_figure_1.png
    ├── week2_figure_2.png
    └── week2_figure_3.png

