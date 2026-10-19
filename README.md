# Official-Media Framing and User Discussions of Marriage in China

**Course:** SOSC 314  
**Group members:** Ruizhe (Rachel) Wang, Jingwen Zhai, and Congxi Guan

## Research Question

**How do Chinese official media frame marriage-related issues, and how does this framing differ from the topics and sentiment expressed in user-initiated online discussions?**

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

For each article, the dataset retains:

- `media`: media outlet
- `date`: publication date
- `title`: article title
- `text`: full article text
- `url`: original article URL
- `matched_keywords`: keywords matched during data collection
- `relevance_label`: initial relevance-screening result

### User-Initiated Weibo Corpus

The second corpus contains original public Weibo posts independently initiated by users.

We collect these posts using the open-source [dataabc/weibo-search](https://github.com/dataabc/weibo-search) scraper together with project-specific Python scripts. Searches are divided by keyword and time period to improve coverage and make the collection process reproducible.

For each Weibo post, the dataset retains available variables such as:

- post ID
- user ID
- post text
- publication date and time
- search keyword
- available engagement indicators

### Time Period and Keywords

The target observation period is 2021–2025. 
The keyword list includes direct marriage terms and related relationship issues:

- **Marriage:** `婚姻`, `结婚`, `婚姻登记`, `结婚登记`, `婚俗`
- **Relationships:** `婚恋`, `恋爱`, `相亲`, `对象`, `伴侣`
- **Marriage status and attitudes:** `单身`, `不婚`, `晚婚`, `恐婚`, `催婚`
- **Marriage-related social issues:** `彩礼`, `高价彩礼`, `离婚`, `离婚冷静期`

### Unit of Analysis

The project uses two document-level units of analysis:

- **Official-media corpus:** one full official-media article
- **User-initiated corpus:** one original Weibo post

### Inclusion and Exclusion Criteria

A document is included when it:

- was publicly available on the selected media website or Weibo;
- was published within the observation period;
- contains at least one predefined marriage-related keyword;
- substantively discusses marriage or a closely related issue;
- contains sufficient usable text for analysis;
- belongs to one of the two defined corpora.

The project excludes:

- duplicate documents;
- advertisements and commercial marketing;
- spam or obvious automated content;
- empty or inaccessible documents;
- keyword matches unrelated to marriage;
- documents containing only incidental marriage references;
- records without sufficient text for analysis.

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
