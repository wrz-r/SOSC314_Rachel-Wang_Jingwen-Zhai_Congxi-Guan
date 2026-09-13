import pandas as pd
import re

# ============================================================
# 1. FILE SETTINGS
# ============================================================

INPUT_FILE = "official_media_articles_2022.csv"

OUTPUT_FILE = "official_media_articles_2022_marriage_cleaned.csv"

REVIEW_FILE = "official_media_articles_2022_object_review.csv"


# ============================================================
# 2. MARRIAGE KEYWORDS
# ============================================================

KEYWORD_GROUPS = {

    "marriage": [

        "婚姻",
        "结婚",
        "婚恋",
        "恋爱",
        "对象",
        "伴侣",
        "相亲",
        "单身",
        "不婚",
        "晚婚",
        "恐婚",
        "催婚",
        "婚姻登记",
        "彩礼",
        "离婚冷静期",

    ]
}


MARRIAGE_KEYWORDS = KEYWORD_GROUPS["marriage"]


# ============================================================
# 3. READ DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ORIGINAL DATA")
print("=" * 60)

print("Number of articles:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 4. MAKE SURE TEXT COLUMN EXISTS
# ============================================================

if "text" not in df.columns:
    raise ValueError(
        "Cannot find column 'text'. "
        "Please check your CSV column names."
    )


df["text"] = df["text"].fillna("").astype(str)


# ============================================================
# 5. OBJECT = PARTNER OR NOT?
# ============================================================

# These patterns strongly suggest that "对象" does NOT mean
# romantic partner.

NON_ROMANTIC_OBJECT_PATTERNS = [

    # research / academic
    r"研究对象",
    r"研究的对象",
    r"调查对象",
    r"调查的对象",
    r"实验对象",
    r"实验的对象",
    r"观察对象",
    r"观察的对象",

    # government / social policy
    r"服务对象",
    r"帮扶对象",
    r"扶助对象",
    r"扶贫对象",
    r"援助对象",
    r"救助对象",
    r"救济对象",
    r"保障对象",
    r"资助对象",
    r"补贴对象",
    r"优抚对象",
    r"低保对象",

    # administration / regulation
    r"管理对象",
    r"监管对象",
    r"监管的对象",
    r"监督对象",
    r"治理对象",
    r"处罚对象",
    r"处罚的对象",
    r"制裁对象",
    r"制裁的对象",
    r"打击对象",
    r"防控对象",
    r"矫正对象",
    r"保护对象",
    r"管控对象",

    # crime / violence
    r"犯罪对象",
    r"犯罪的对象",
    r"攻击对象",
    r"袭击对象",

    # media / work
    r"采访对象",
    r"采访的对象",
    r"报道对象",
    r"报道的对象",
    r"拍摄对象",
    r"拍摄的对象",
    r"写作对象",

    # business / organizations
    r"合作对象",
    r"竞争对象",
    r"客户对象",
    r"消费者对象",
    r"服务对象",

    # geographic / institutional
    r"对象国",
    r"对象国家",
    r"对象地区",
    r"对象企业",
    r"对象公司",
    r"对象人群",
    r"对象群体",
    r"对象用户",

    # generic expressions
    r"对象包括",
    r"对象主要",
    r"对象之一",
    r"对象数量",
    r"对象名单",
    r"对象范围",
    r"对象标准",
    r"对象信息",
    r"对象身份",
    r"对象资格",
    r"对象分类",

    # other common non-romantic uses
    r"崇拜对象",
    r"模仿对象",
    r"学习对象",
    r"表彰对象",
    r"先进对象",
    r"狩猎对象",
    r"比较对象",
    r"参照对象",
]


# These patterns strongly suggest that "对象" DOES mean
# romantic partner.

ROMANTIC_OBJECT_PATTERNS = [

    r"找对象",
    r"找个对象",
    r"找一个对象",

    r"介绍对象",
    r"介绍个对象",
    r"介绍一个对象",

    r"相亲对象",
    r"结婚对象",
    r"恋爱对象",
    r"婚恋对象",
    r"择偶对象",

    r"合适对象",
    r"合适的对象",

    r"没有对象",
    r"没对象",
    r"有对象",

    r"谈对象",
    r"谈了对象",
    r"谈个对象",

    r"处对象",
    r"处了对象",

    r"对象匹配",
    r"对象推荐",
    r"对象条件",
    r"对象要求",

    r"和对象",
    r"跟对象",
    r"与对象",

    r"对象父母",
    r"对象家庭",
    r"对象家",

]


def classify_object(text):

    if "对象" not in text:
        return "no_object"

    # --------------------------------------------------------
    # First: clearly romantic
    # --------------------------------------------------------

    for pattern in ROMANTIC_OBJECT_PATTERNS:

        if re.search(pattern, text):

            return "romantic"


    # --------------------------------------------------------
    # Second: clearly non-romantic
    # --------------------------------------------------------

    for pattern in NON_ROMANTIC_OBJECT_PATTERNS:

        if re.search(pattern, text):

            return "non_romantic"


    # --------------------------------------------------------
    # Third: unclear
    # --------------------------------------------------------

    return "ambiguous"


df["object_class"] = df["text"].apply(classify_object)


# ============================================================
# 6. SHOW OBJECT CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("OBJECT CLASSIFICATION")
print("=" * 60)

print(
    df["object_class"].value_counts()
)


# ============================================================
# 7. CLEAN MATCHED KEYWORDS
# ============================================================

def clean_matched_keywords(row):

    original = str(row.get("matched_keywords", ""))

    # Your scraper may separate keywords with |.
    keywords = [
        x.strip()
        for x in original.split("|")
        if x.strip()
    ]

    cleaned = []

    for keyword in keywords:

        # If keyword is "对象", only keep it when it is
        # romantic or ambiguous.
        #
        # Non-romantic "对象" should NOT count as a
        # marriage keyword.

        if keyword == "对象":

            if row["object_class"] == "non_romantic":
                continue

        cleaned.append(keyword)

    return "|".join(cleaned)


df["matched_keywords_clean"] = df.apply(
    clean_matched_keywords,
    axis=1
)


# ============================================================
# 8. REMOVE ARTICLES WITH NO VALID MARRIAGE KEYWORDS
# ============================================================

df_clean = df[
    df["matched_keywords_clean"].str.strip() != ""
].copy()


# ============================================================
# 9. UPDATE KEYWORD COLUMN
# ============================================================

df_clean["matched_keywords"] = df_clean[
    "matched_keywords_clean"
]


# ============================================================
# 10. CREATE / UPDATE KEYWORD GROUP
# ============================================================

df_clean["keyword_group"] = "marriage"


# ============================================================
# 11. SAVE AMBIGUOUS "对象" ARTICLES
# ============================================================

review = df[
    df["object_class"] == "ambiguous"
].copy()


review_columns = [
    col
    for col in [
        "media",
        "date",
        "title",
        "text",
        "matched_keywords",
        "object_class"
    ]
    if col in review.columns
]


review[review_columns].to_csv(
    REVIEW_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. DROP TEMPORARY COLUMNS
# ============================================================

df_clean.drop(
    columns=["matched_keywords_clean"],
    inplace=True
)


# ============================================================
# 13. SAVE FINAL DATA
# ============================================================

df_clean.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(
    f"Original articles: {len(df)}"
)

print(
    f"Cleaned articles:  {len(df_clean)}"
)

print(
    f"Removed articles:  {len(df) - len(df_clean)}"
)

print("\nObject classification:")
print(
    df["object_class"].value_counts()
)

print("\nFiles created:")

print(
    f"1. {OUTPUT_FILE}"
)

print(
    f"2. {REVIEW_FILE}"
)

print("\nDone.")