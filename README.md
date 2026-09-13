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

## Time Period and Keywords

The target observation period is 2021–2025. 
The keyword list includes direct marriage terms and related relationship issues:

- **Marriage:** `婚姻`, `结婚`, `婚姻登记`, `结婚登记`, `婚俗`
- **Relationships:** `婚恋`, `恋爱`, `相亲`, `对象`, `伴侣`
- **Marriage status and attitudes:** `单身`, `不婚`, `晚婚`, `恐婚`, `催婚`
- **Marriage-related social issues:** `彩礼`, `高价彩礼`, `离婚`, `离婚冷静期`

## Unit of Analysis

The project uses two document-level units of analysis:

- **Official-media corpus:** one full official-media article
- **User-initiated corpus:** one original Weibo post

## Inclusion and Exclusion Criteria

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
