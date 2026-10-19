# Official-Media Framing and User Discussions of Marriage in China

**Course:** SOSC 314  
**Group members:** Ruizhe (Rachel) Wang, Jingwen Zhai, and Congxi Guan

## Research Question

**How do Chinese official media frame marriage, and how do their topics and attitudes differ from user-initiated Weibo discussions?**

## Project Purpose and Background

Marriage has become an increasingly important topic in Chinese public discussion. Official media and ordinary social-media users may emphasize different aspects of these issues. Official media may discuss marriage through policy, demographic, legal, or social perspectives, while users may focus more on personal experiences, financial pressure, relationships...

This project compares marriage-related discourse in Chinese official media with discussions independently initiated by Weibo users. The purpose is to examine whether institutional media representations of marriage are similar to or different from the topics and sentiment expressed in public online discussions.

## Data Sources

The study consists of two Chinese-language text corpora:

1. Full-text articles from Chinese official-media websites
2. Original marriage-related posts initiated by Weibo users

The target observation period is **January 1, 2021 to December 31, 2025**. 

### Official-Media Article Corpus

The official-media corpus contains full-text articles from three major Chinese official-media sources:

- [People.cn](http://people.cn)
- CCTV
- China News Service

Articles are identified through keyword-based searches of the selected media websites. After candidate articles are identified, the full text is collected from the original news webpages.

### User-Initiated Weibo Corpus

The second corpus contains original public Weibo posts independently initiated by users.

We collect these posts using the open-source [dataabc/weibo-search](https://github.com/dataabc/weibo-search) scraper together with project-specific Python scripts. Searches are divided by keyword and time period to improve coverage and make the collection process reproducible.


### Time Period and Keywords

The target observation period is 2021–2025. 
The keyword list includes direct marriage terms and related relationship issues:

- **Marriage:** `婚姻`, `结婚`, `婚姻登记`, `结婚登记`, `婚俗`
- **Relationships:** `婚恋`, `恋爱`, `相亲`, `对象`, `伴侣`
- **Marriage status and attitudes:** `单身`, `不婚`, `晚婚`, `恐婚`, `催婚`
- **Marriage-related social issues:** `彩礼`, `高价彩礼`, `离婚`, `离婚冷静期`

**Official media**  
**5,265** full-text articles  
People.cn, CCTV, and China News Service  
Unit: one article

**Weibo**  
**41,568** cleaned, relevant, deduplicated posts  
Original public posts collected with [dataabc/weibo-search](https://github.com/dataabc/weibo-search)  
Unit: one post

| Dimension | Official media | Weibo |
| --- | --- | --- |
| **Provenance** | Official search results and article pages | Public search results from a pinned scraper version |
| **Core fields** | Source, date, title, text, URL, matched keywords | Time block, query keyword, post text, publication time |
| **Screening** | Matched terms and relevance class | Relevance level, rule, and matched evidence |
| **Model inputs** | Year, cleaned text, tokens, and token counts | |


### Cleaning, Relevance, and Preprocessing

| **Stage** | **Official-media articles** | **Weibo posts** |
|---|---|---|
| **Quality** | Remove missing, short, or invalid pages. | Remove blank/non-Chinese posts and bodies lacking the query term. |
| **Noise** | Remove boilerplate and extraction artifacts. | Exclude fiction, platform, fandom/game, celebrity, and other-language indicators. |
| **Relevance** | Require valid marriage terms; screen ambiguous uses. | Accept strong terms; require context, first-person attitude, or multiple terms for weak matches. |
| **Duplicates** | Remove duplicate URLs, titles, and text. | For identical ordered Chinese-character sequences, retain the earliest post. |
| **NLP** | Tokenize with `jieba`; combine public and project stopwords; protect marriage terms; require ≥10 article tokens or ≥5 post tokens. | 


## Research Method

The project uses comparative computational text analysis to examine official-media framing and user-initiated online discussion.

The main research process is:

1. Search official-media websites for marriage-related articles using predefined keywords.
2. Retrieve the full text of relevant official-media articles.
3. Search Weibo by keyword and time period using [dataabc/weibo-search](https://github.com/dataabc/weibo-search).
4. Retain original user-initiated Weibo posts related to marriage.
5. Identify the major topics and frames in official-media articles.
6. Identify the major topics and sentiment expressed in user-initiated posts.
7. Compare the two corpora and examine changes across media sources and over time.

### Methods-Topic Modeling
#### Fightin' Words
We first used **Fightin' Words**, a weighted log-odds method, to identify vocabulary that is more characteristic of official-media articles or Weibo posts. Positive z-scores indicate words more distinctive of official media; negative z-scores indicate words more distinctive of Weibo. This initial comparison identifies which words stand out in each corpus and provides a first indication of differences in emphasis.

#### LDA Topic Modeling
We then applied **Latent Dirichlet Allocation (LDA)**, an unsupervised machine learning method that identifies topics from patterns of words appearing together. We fitted **separate models** to the official-media and Weibo corpora, then examined their topics, average prevalence, and yearly changes.

**LDA setup and model selection**

- **Text representation:** Bag-of-words (word counts), with a maximum vocabulary of 5,000 words.
- **Training:** Online LDA; random seed 42 for the main models.
- **Baseline:** K = 8 topics. We also tested K = 10, 12, 15, 20, and 25.
- **Selection:**  We compared interpretability, topic specificity, and simplicity, aiming to avoid topics that were too broad or unnecessarily fragmented.
- **Final choice**: K = 12 for both corpora.

### Results
#### Fightin' Words: Distinctive Vocabulary
The strongest official-media terms included marriage registration (z = +58.8), bride price (+45.0), application processing (+35.1), and services (+35.1). These terms point to a focus on administrative procedures, legal issues, and public services.

Weibo-distinctive terms included getting married (z = −44.6), liking (−40.7), children (−37.0), and pressure to marry (−35.3). They suggest greater attention to personal and family experiences.

Fightin' Words offers initial evidence of different emphases, but the Weibo results contain broad conversational words such as liking and feelings, which do not identify clear marriage-related topics on their own. We therefore used LDA to examine the broader themes.

#### LDA Topics

##### Table 1. Official-Media Topics (\(K = 12\))

| **Topic** | **Interpretation** | **Representative Keywords** | **Mean Topic Share** |
|:---|:---|:---|---:|
| T1 | Bride-price returns | bride price, return, life, payment, circumstances | 5.1% |
| T2 | Marriage choices | marriage, marry, family, life, parents | 10.5% |
| T3 | Fertility policy | childbearing, marriage registration, policy, marry, registration | 4.5% |
| T4 | Marriage registration | marriage registration, processing, marriage registration, newlyweds, services | 17.7% |
| T5 | Rural bride-price reform | bride price, rural, custom reform, high bride price, governance | 9.7% |
| T6 | Wedding reform | civility, wedding, events, marriage customs, newlyweds | 15.8% |
| T7 | Youth dating | young adults, events, matchmaking, dating, romance and marriage | 8.3% |
| T8 | Divorce disputes | divorce, court, children, property, spouses | 8.6% |
| T9 | Domestic violence | domestic violence, personal safety protection order, domestic abuse, divorce, protection | 1.9% |
| T10 | Matchmaking services | services, platform, matchmaking, marry, matchmaking agency | 7.1% |
| T11 | Romance scams | fraud, programs, crime, police officers, older adults | 4.8% |
| T12 | Family mediation | mediation, family, work, disputes, marriage and family | 5.8% |

##### Table 2. Weibo Topics (\(K = 12\))

| **Topic** | **Interpretation** | **Representative Keywords** | **Mean Topic Share** |
|:---|:---|:---|---:|
| T1 | Marriage choices | marriage, life, life course, choice, marry | 11.4% |
| T2 | Gender and fertility | women, society, marriage, romance and marriage, family | 8.0% |
| T3 | Dating feelings | liking, dating, dating, love, feelings | 8.3% |
| T4 | Bride price | bride price, men, women, women, men | 6.5% |
| T5 | Marriage hopes | happiness, hope, marry, friends, happiness | 10.9% |
| T6 | Marriage pressure | pressure to marry, non-marriage, matchmaking, late marriage, my mother | 7.3% |
| T7 | Relationship feelings | relationships, emotions, feelings, partner, parents | 8.3% |
| T8 | Daily life (mixed) | evening, phone, single, tomorrow, sleep | 5.5% |
| T9 | Matchmaking (mixed) | feelings, matchmaking, eating, evening, teacher | 7.6% |
| T10 | Divorce | divorce, divorce cooling-off period, marry, marriage, marriage registration | 5.6% |
| T11 | Work and marriage | work, friends, marry, one year, life | 7.1% |
| T12 | Children and family | marry, children, mother, fear of marriage, parents | 13.4% |

#### Topic Prevalence
The most prevalent topics in official media were **Marriage registration (17.7%)** and **Wedding reform (15.8%)**, followed by Marriage choices (10.5%) and Rural bride-price reform (9.7%). Together, these topics suggest that official media gave considerable attention to marriage registration services and efforts to change wedding customs.

On Weibo, the largest themes were **Children and family (13.4%)**, **Marriage choices (11.4%)**, and **Marriage hopes (10.9%)**, alongside Dating feelings (8.3%) and Relationship feelings (8.3%). Family responsibilities, personal decisions, and expectations about relationships were recurring parts of online discussion. Importantly, Marriage hopes and Dating feelings show that user discussions included aspirations and relationship experiences, rather than only marriage-related pressures.

The topic distributions show **two different ways of discussing marriage**. Official-media articles give more attention to how marriage is managed and how wedding customs can be changed. Weibo discussions focus more on family life, the decision to marry, and expectations about relationships. This suggests that official media often presents marriage as a matter of public policy and social practices, while users more often discuss its place in their personal and family lives.



#### Shared Issues: Bride Price and Divorce

Interestingly, **bride price and divorce** appeared as topics in both corpora. This indicates that they are recurring concerns in both official media and Weibo discussions. To understand how the two sources approach the same issue, we compared the **weights of selected words within the corresponding LDA topics**.

**Bride price** (money or gifts traditionally given by the groom's family to the bride's family as part of marriage arrangements) appeared in both corpora. In the official-media topic, *return* had a weight of **3.34%**, compared with less than **0.01%** in the Weibo topic. Together with words about payment and courts, this suggests an emphasis on disputes about bride-price payments and how they are resolved. In the Weibo topic, *men* (**6.07%**) and *women* (**4.60%**) were more prominent, compared with less than **0.01%** for each word in the selected official-media topic. These terms suggest more attention to the expectations placed on men and women when preparing for marriage. In short, official-media vocabulary points more toward **handling disputes**, while Weibo vocabulary points more toward **gender relations and marriage expectations**.

*Figure 4. Selected keyword weights in the official-media and Weibo bride-price topics.*

**Divorce.** The official-media divorce topic gave more weight to *court* (**1.96%**, compared with **0.58%** on Weibo) and *judgment* (**0.87%**, compared with **0.16%**). This suggests greater attention to legal cases and decisions made by courts. The Weibo topic emphasized the *divorce cooling-off period* (a waiting period before a couple can complete a divorce by mutual agreement through registration): its weight was **5.05%**, compared with **0.22%** in official media. *Domestic violence* was also more prominent in the Weibo topic (**1.62%**, compared with less than **0.01%**). These terms suggest that Weibo discussions give more attention to the difficulties people may face during divorce and to safety within marriage.

*Figure 5. Selected keyword weights in the official-media and Weibo divorce topics.*

#### Topic Trends, 2021–2025

The clearest change in official media was a shift **from discussing young people's marriage choices toward reforms of wedding customs**. Among the topics shown in Figure 5:

- **Marriage choices** decreased from **18.3% (2021)** to **4.2% (2025)**.
- **Wedding reform** increased from **7.9%** to **19.4%**.
- **Rural bride-price reform** increased from **4.8%** to **10.0%**, with a peak of **12.4% in 2024**.
- **Fertility policy** declined from **8.7%** to **2.8%**.

These changes may be connected to broader developments in China. Marriage registration numbers have declined over the longer term, while the **cost of marriage**, especially high bride prices, has attracted public and policy attention. Government initiatives to discourage expensive wedding customs and address bride-price disputes also became more prominent during this period. For example, marriage-registration and wedding-custom reforms continued in 2025. This context may help explain the growing attention to wedding and bride-price reform in official media.

*Figure 6. Selected official-media topic trends, 2021–2025.*

Weibo discussions showed a different pattern. The shares of major topics changed relatively little between 2021 and 2025.

Taken together, the trends suggest that **official-media priorities changed more noticeably, while the main concerns expressed on Weibo were relatively consistent**. A possible explanation is that official media increasingly discussed marriage in connection with social and policy efforts to reduce the costs and change the customs of marriage, while Weibo users continued to talk about how marriage affects their everyday lives.

*Figure 7. Trends in Weibo topic prevalence, 2021–2025.*

### LDA Topic Model Diagnostics and Robustness

To check how reliable the **LDA topic results** were, we examined topic coherence and exclusivity, random-seed stability, and sensitivity to the number of topics (K).

| Diagnostic | Purpose | Main observation |
| --- | --- | --- |
| **Coherence and exclusivity** | Check whether the top words form meaningful, distinctive topics. | Several topics are clear, but some broad themes have weaker distinctiveness. |
| **Random-seed stability** | Compare topic solutions from different initializations with the main model (seed 42). | Official-media topic similarity ranged from **0.74 to 0.82**; Weibo ranged from **0.62 to 0.75** across the displayed seeds. |
| **Sensitivity to K** | Check whether topics persist when K changes. | Major themes are recognizable across settings, although some topics split or merge. |

#### Coherence and Exclusivity

*Figure 8. Topic coherence and exclusivity for the K = 12 models.*

#### Random-Seed Stability

Topic similarity was generally higher for the official-media models than for the Weibo models, suggesting that the Weibo topic solution is more sensitive to initialization.

*Figure 9. Random-seed stability relative to seed 42 at K = 12.*

#### Sensitivity to the Number of Topics

We also compared LDA results using different numbers of topics. The broader themes were still recognizable, but some specific topics were split into smaller ones or combined with other topics as K changed. We chose **K = 12** because it provided a useful balance of clear topics and manageable detail.

#### Topic-Alignment Diagnostic

To compare alternative K values with the K = 12 baseline, Hungarian matching ([Munkres, 1957](https://doi.org/10.1137/0105003)) pairs topics by maximum cosine similarity of their word-weight vectors. Cosine similarity, top-word Jaccard overlap, and yearly-prevalence Spearman correlation then measure content and trend stability without confusing topic-label switching with model change. Figure 10 reports this comparison.

*Figure 10. Topic similarity when the number of topics K is changed.*

**Interpretation.** Broad themes remain recognizable when *K* changes, but narrower topics can split or merge, especially on Weibo. Conclusions therefore emphasize aggregate differences rather than one-to-one replication of every topic.

### Interpretation and Substantive Takeaways

The main finding is that **official media and Weibo users place different priorities on marriage-related issues**. Official-media articles devote more attention to marriage registration, legal processes, and reforms of wedding customs. Weibo users more often discuss family responsibilities, relationships, pressure from others, and the personal decision to marry. This difference matters because it shows which aspects of marriage become visible in official-media articles and which concerns are more prominent in everyday online conversations.

The comparisons of **bride price and divorce** add a further insight. Even when both sources discuss the same issue, they may describe the problem differently. For example, discussions focused on courts or the return of bride-price payments points toward **legal rules and institutional solutions**. Discussions focused on gender expectations, divorce waiting periods, or domestic violence raise questions about **fairness, personal choices, and lived experiences**.

The trends also suggest that the relationship between these discussions is **not fixed over time**. Official-media attention shifted toward wedding and bride-price reform, while the main themes on Weibo changed relatively little. Together, these results point to a **possible gap between institutional priorities and concerns expressed in everyday discussion**. Understanding that gap can help researchers ask whether public communication addresses the aspects of marriage that people find most relevant. Our analysis identifies the difference in emphasis; it cannot determine whether official media is responding to, or overlooking, specific public concerns.

### Limitations and Future Work

- **Corpus differences:** Official articles and short social-media posts differ in length and writing style, which can affect word and topic distributions.
- **Topic uncertainty:** Some topics are broad or overlapping, especially on Weibo. Topic labels involve researcher judgment, and the results vary somewhat with random seeds and K.
- **Future work:** Reading representative posts and articles, or comparing discussions around specific events, could help test how shared issues are framed and whether official media responds to public concerns.

---

## Methods-Attitude Analysis

Topic modeling tells us what official media and users are talking about, but it does not tell us whether marriage is being represented positively, negatively, or neutrally. We therefore introduce attitude analysis to help us evaluate the sentiment towards marriage.

We compare **Semantic Scaling** with a supervised **Transformer classifier**.

### Semantic Scaling

The model embeds documents and two sets of anchor words using the Chinese model `BAAI/bge-small-zh-v1.5` ([Xiao et al., 2023](https://arxiv.org/abs/2309.07597)).

- **Positive anchors:** 幸福、美好、和谐、陪伴、稳定 *(happiness, harmony, companionship, stability)*
- **negative anchors:** 压力、负担、痛苦、束缚、恐惧 *(pressure, burden, pain, confinement, fear)*

Each document receives a continuous score based on its relative proximity to the two semantic poles and calculate *ScoreS*

  

### Transformer-Based Attitude Classification

After developing the attitude codebook through human annotation of 100 posts, we trained a supervised Transformer-based classifier. The purpose of the model is to learn the relationship between the linguistic context of a document and its human-coded attitude, rather than relying on the presence or frequency of individual keywords.

We used `hfl/chinese-roberta-wwm-ext` ([Cui et al., 2020](https://doi.org/10.18653/v1/2020.findings-emnlp.58)), a Chinese RoBERTa model with the three attitude categories defined in our codebook: negative (-1), neutral (0), and positive (+1).

We calculate:

_ScoreT = P(+1) − P(−1)_

  

We prioritize the **Transformer** because its categories correspond directly to the human annotation scheme and can be evaluated within the context.
It also achieved **82.6% validation accuracy**. Across 3,935 documents, its scores had moderate positive correlations with Semantic Scaling (Pearson *r* = 0.382; Spearman *ρ* = 0.370). On the manually annotated subset, the correlations were *r* = 0.472 and *ρ* = 0.504.

*Figure 11. Distributions of Transformer and Semantic Scaling attitude scores.*

### Evaluation and results

The results show a clear difference in marriage attitudes between official media and user-initiated posts. Overall, official media posts are more positive toward marriage than user posts.

| Result | Official media | User-initiated posts |
| --- | --- | --- |
| Mean Transformer score | +0.31 | -0.03 |
| Predicted positive | 56% | 20% |
| Predicted neutral | 42% | 27% |
| Predicted negative | approximately 2% | 54% |

*Figure 12. Distribution of marriage attitudes in official-media and user-initiated posts.*

From 2021 to 2025, official-media scores fluctuate, rising from about +0.26 in 2021 to +0.39 in 2025, with a decline in 2024. User scores remain close to or below zero. The estimated official-user gap is positive in every year and largest in 2025.

*Figure 13. Attitude trends over five years.*

The categorical results further suggest that this difference is driven primarily by the relative absence of negative attitudes in official media rather than by a much larger share of positive attitudes. Positive texts account for 56% of official-media posts while 54% of user posts are classified as negative.

*Figure 14. Attitude composition of official-media and user-initiated posts.*

Overall, these results indicate a persistent divergence between institutional and user-initiated marriage discourse. **Official media tend to frame marriage more positively and show greater variation across time, whereas user attitudes remain relatively stable and mildly positive.** The divergence becomes particularly pronounced in 2025.

### Diagnostics and robustness

#### Sensitivity Analysis

A sensitivity analysis compared **five preprocessing specifications**: baseline cleaning, raw text, retaining hashtags, and maximum sequence lengths of 128 and 256 tokens. The consistency of the results indicates that the substantive finding is robust to reasonable preprocessing choices, rather than being an artifact of a particular text-cleaning or truncation procedure.

*Figure 15. Sensitivity of the Official–Weibo attitude gap across specifications.*

#### Seed Stability

Two better-performing random-seed runs, with validation macro-F1 scores of 0.725 and 0.692, produced positive gaps, while three weaker runs produced negative or near-zero gaps. This indicates that model performance is sensitive to random initialization given the small labelled sample, and that the substantive gap is conditional on adequate model performance.

*Figure 16. Attitude-model stability across random seeds.*

A manual audit of 40 Weibo posts found 90% human-model agreement among high-confidence cases and 50% among low-confidence cases. The model misclassified 11 of 12 human-labelled negative posts as neutral or positive. This suggests that it under-detects negative attitudes, especially in ironic or self-deprecating expressions.

*Figure 17. Human–model agreement in the Weibo audit sample.*

#### Semantic Scaling Robustness

Holding BAAI/bge-small-zh-v1.5 fixed ([Xiao et al., 2023](https://arxiv.org/abs/2309.07597)), we vary: (1) one positive–negative anchor pair at a time ([Efron & Tibshirani, 1993](https://doi.org/10.1007/978-1-4899-4541-9)), (2) MIN\_FREQ = 2, 3, or 5, and (3) mean, median, or trimmed-mean aggregation. Alternatives are compared with the baseline using document-score and yearly-profile Spearman correlations plus the signs of the official-media mean and Official–Weibo gap.

*Figure 18. Semantic Scaling robustness across internal settings.*

Document scores are mostly stable: the anchor-pair jackknife median is 0.954, MIN\_FREQ alternatives exceed 0.99, and the trimmed mean reaches 0.976. Median aggregation is weaker (0.827). Both corpus means remain positive, although magnitudes and annual profiles vary.

*Figure 19. Semantic Scaling robustness assessment using predefined descriptive thresholds.*

Vocabulary frequency passes with correlations near 0.99. Anchor-pair omission is partial because the minimum yearly-profile correlation is −0.36 and gap-sign stability is 68%; aggregation is also partial. The overall positive direction is robust, but yearly patterns and the exact gap require caution. Thresholds are descriptive, not significance tests.

### Interpretation and limitations

The results suggest that official media portray marriage more positively than user-initiated posts, with a difference that persists throughout 2021–2025.

Measurement remains the main limitation. The **initial labelled sample is small, negative Weibo posts are difficult to classify** due to the language. The result shown before is the re-trained model after the 40-post-manual audit, but bias still exists.

Next steps should expand the labelled dataset, include more negative examples, and report cross-validation plus class-specific precision, recall, and macro-F1. A larger manual audit stratified by source and predicted category would help locate systematic errors.

#### Additional Takeaway

The strongest conclusion is aggregate: official media emphasize institutional and policy concerns and portray marriage more positively, while Weibo foregrounds personal pressure and lived experience. The results do not show causal media effects or population attitudes.

#### Additional Limitation and Next Step

Individual topics, annual attitude profiles, and the exact cross-corpus gap are less stable than the overall direction. Future work should expand stratified human labels—especially ironic, implicit, and negative posts—and add event-matched qualitative validation.


## Repository Structure

```text
.
├── README.md
├── literature review
├── data
│   ├── official-media articles
│   ├── user-initiated Weibo posts
│   └── pilot data
├── scripts
│   ├── official-media data collection and processing
│   ├── user-initiated Weibo data collection and processing
│   └── data visualization
└── figures
    ├── official-media figures
    ├── user-discussion figures
    └── pilot-study figures
```
