# Install the Unicode-aware regex package and upload exactly ten CSV files.
import io
import json
import subprocess
import sys
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "regex"],
    check=True,
)

import pandas as pd
import regex
from google.colab import files

input_files = []
NUMBER_OF_FILES = 10
for upload_number in range(1, NUMBER_OF_FILES + 1):
    print(f"Upload CSV file {upload_number} of {NUMBER_OF_FILES}:")
    one_upload = files.upload()
    uploaded_csvs = [
        (name, data) for name, data in one_upload.items()
        if name.lower().endswith(".csv")
    ]
    if len(uploaded_csvs) != 1:
        raise ValueError(
            f"Upload exactly one CSV in upload window {upload_number}; "
            f"received {[name for name, _ in uploaded_csvs]}"
        )
    filename, file_bytes = uploaded_csvs[0]
    input_files.append((upload_number, filename, file_bytes))

OUTPUT_DIR = Path("/content/weibo_10_candidate_files_merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Input CSV files:", [name for _, name, _ in input_files])



# Read and merge all ten files while preserving source file and upload order.
frames = []
for upload_number, filename, file_bytes in input_files:
    frame = pd.read_csv(
        io.BytesIO(file_bytes),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    frame["source_file"] = filename
    frame["upload_number"] = upload_number
    frame["source_row"] = range(2, len(frame) + 2)  # CSV row number including header
    frames.append(frame)
    print(filename, "rows:", len(frame))

merged = pd.concat(frames, ignore_index=True, sort=False).fillna("")
required = {"微博正文", "发布时间"}
missing = required - set(merged.columns)
if missing:
    raise ValueError(f"Required columns are missing: {sorted(missing)}")

MERGED_FILE = OUTPUT_DIR / "eligible_candidates_merged_before_cleaning.csv"
merged.to_csv(MERGED_FILE, index=False, encoding="utf-8-sig")
print("Merged rows:", len(merged))
print("Merged backup:", MERGED_FILE)



# Reapply the same content-cleaning rules used in the latest crawler notebook.
FICTION_TERMS = ["小说", "阅读全文", "完结", "大结局", "晋江文学", "txt", "全章节"]
EXCLUDE_TERMS = ["微博问答", "超话", "恋与深空", "恋与制作人", "微博正文"]
CELEBRITY_TERMS = [
    "谢娜", "何炅", "杨幂", "Angelababy", "陈坤", "赵丽颖", "易烊千玺", "王源", "王俊凯",
    "姚晨", "张杰", "迪丽热巴", "唐嫣", "林心如", "邓超", "陈乔恩", "刘亦菲", "杨紫",
    "宋茜", "赵薇", "郭德纲", "林志颖", "胡歌", "范冰冰", "王力宏", "陈赫", "黄子韬",
    "鹿晗", "贾乃亮", "罗志祥", "黄晓明", "薛之谦", "杨洋", "李晨", "林俊杰", "韩庚",
    "林更新", "刘烨", "张艺兴", "高圆圆", "刘涛", "孙俪", "王珞丹", "佟丽娅", "关晓彤",
    "吴磊", "戚薇", "范玮琪", "郑恺", "王祖蓝", "马伊琍", "张靓颖", "蔡依林", "赵露思",
    "郭采洁", "周冬雨", "虞书欣", "章子怡", "白鹿", "李沁", "景甜", "倪妮", "张馨予",
    "白百何", "古力娜扎", "江疏影", "杨超越", "肖战", "王一博", "马嘉祺", "宋亚轩",
    "刘耀文", "丁程鑫", "严浩翔", "张真源", "贺峻霖",
]

fiction_pattern = regex.compile(
    "|".join(map(regex.escape, FICTION_TERMS)), regex.IGNORECASE
)
exclude_pattern = regex.compile("|".join(map(regex.escape, EXCLUDE_TERMS)))
celebrity_pattern = regex.compile(
    "|".join(map(regex.escape, CELEBRITY_TERMS)), regex.IGNORECASE
)
other_script_pattern = regex.compile(
    r"(?!\p{Script=Han}|\p{Script=Latin})\p{L}"
)
han_character_pattern = regex.compile(r"\p{Script=Han}")

def find_cleaning_reasons(row):
    body = str(row.get("微博正文", "")).strip()
    reasons = []
    if not body:
        reasons.append("blank_body")

    query_keyword = str(row.get("query_keyword", "")).strip()
    if query_keyword and query_keyword not in body:
        reasons.append("query_keyword_not_in_body:" + query_keyword)

    fiction = fiction_pattern.search(body)
    excluded = exclude_pattern.search(body)
    celebrity = celebrity_pattern.search(body)
    other_script = other_script_pattern.search(body)
    if fiction:
        reasons.append("fiction_term:" + fiction.group())
    if excluded:
        reasons.append("excluded_term:" + excluded.group())
    if celebrity:
        reasons.append("celebrity_name:" + celebrity.group())
    if other_script:
        reasons.append("other_script:" + other_script.group())
    if body and not han_character_pattern.search(body):
        reasons.append("no_han_characters")
    return "; ".join(reasons)

merged["cleaning_removal_reason"] = merged.apply(find_cleaning_reasons, axis=1)
cleaning_removed = merged[merged["cleaning_removal_reason"] != ""].copy()
cleaned = merged[merged["cleaning_removal_reason"] == ""].copy()

CLEANING_REMOVED_FILE = OUTPUT_DIR / "removed_by_cleaning.csv"
CLEANED_BEFORE_RELEVANCE_FILE = OUTPUT_DIR / "cleaned_before_relevance_filter.csv"
cleaning_removed.to_csv(CLEANING_REMOVED_FILE, index=False, encoding="utf-8-sig")
cleaned.to_csv(CLEANED_BEFORE_RELEVANCE_FILE, index=False, encoding="utf-8-sig")

print("Rows before cleaning:", len(merged))
print("Rows removed by cleaning:", len(cleaning_removed))
print("Rows retained before relevance filtering:", len(cleaned))



# Score marriage relevance using the six deterministic rules.
STRONG_KEYWORDS = [
    "婚姻", "结婚", "婚恋", "不婚", "晚婚", "恐婚", "催婚",
    "婚姻登记", "彩礼", "离婚冷静期",
]
WEAK_KEYWORDS = ["恋爱", "对象", "伴侣", "相亲", "单身"]
MARRIAGE_CONTEXT_TERMS = [
    "夫妻", "配偶", "丈夫", "妻子", "老公", "老婆", "男朋友", "女朋友",
    "男友", "女友", "领证", "婚礼", "婚后", "未婚", "已婚", "离婚",
    "再婚", "二婚", "求婚", "择偶", "脱单", "成家", "嫁人", "娶妻",
    "婚姻观", "恋爱观", "亲密关系", "父母催婚", "适婚年龄", "终身大事",
]
FIRST_PERSON_TERMS = ["我", "我们", "本人", "自己"]
ATTITUDE_TERMS = [
    "觉得", "认为", "感觉", "希望", "想要", "不想", "愿意", "不愿意",
    "接受", "不能接受", "喜欢", "不喜欢", "害怕", "焦虑", "支持", "反对",
    "选择", "后悔", "压力", "自由", "幸福", "孤独",
]
AMBIGUOUS_PATTERNS = [
    "研究对象", "实验对象", "调查对象", "服务对象", "适用对象", "目标对象",
    "招生对象", "招聘对象", "保障对象", "救助对象", "犯罪对象", "监测对象",
    "学习伴侣", "旅行伴侣", "智能伴侣", "AI伴侣", "宠物伴侣", "机器人伴侣",
    "单身公寓", "单身套餐", "单身房", "恋爱游戏", "恋爱综艺", "恋爱小说",
    "恋爱剧情", "恋爱漫画",
]
PROXIMITY_LIMIT = 20  # Han characters between a weak term and context term

def unique_hits(text, terms, ignore_case=False):
    haystack = str(text).casefold() if ignore_case else str(text)
    return [
        term for term in terms
        if (term.casefold() if ignore_case else term) in haystack
    ]

def term_spans(text, terms):
    spans = []
    for term in terms:
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            spans.append((index, index + len(term), term))
            start = index + 1
    return spans

def minimum_span_gap(text, left_terms, right_terms):
    left_spans = term_spans(text, left_terms)
    right_spans = term_spans(text, right_terms)
    if not left_spans or not right_spans:
        return None
    return min(
        max(0, max(left_start, right_start) - min(left_end, right_end))
        for left_start, left_end, _ in left_spans
        for right_start, right_end, _ in right_spans
    )

def score_relevance(body):
    original = str(body)
    han_text = "".join(regex.findall(r"\p{Script=Han}", original))
    strong = unique_hits(han_text, STRONG_KEYWORDS)
    weak = unique_hits(han_text, WEAK_KEYWORDS)
    context = unique_hits(han_text, MARRIAGE_CONTEXT_TERMS)
    first_person = unique_hits(han_text, FIRST_PERSON_TERMS)
    attitude = unique_hits(han_text, ATTITUDE_TERMS)
    ambiguous = unique_hits(original, AMBIGUOUS_PATTERNS, ignore_case=True)
    gap = minimum_span_gap(han_text, weak, context)

    # A: any strong keyword is direct evidence of marriage relevance.
    if strong:
        level, rule, reason = "high_relevance", "A_strong_keyword", ""
    # F overrides C/D when there is no strong or marriage-context evidence.
    elif ambiguous and not context:
        level, rule = "irrelevant", "F_fixed_ambiguous_expression"
        reason = "fixed_ambiguity_without_strong_or_context"
    # B: weak keyword and marriage context within 20 Han characters.
    elif weak and context and gap is not None and gap <= PROXIMITY_LIMIT:
        level, rule, reason = "high_relevance", "B_weak_near_context", ""
    # C: weak keyword plus both first-person and attitude expression.
    elif weak and first_person and attitude:
        level, rule, reason = "high_relevance", "C_personal_attitude", ""
    # D: two or more different weak keywords.
    elif len(weak) >= 2:
        level, rule, reason = "high_relevance", "D_multiple_weak_keywords", ""
    # Partial evidence is retained separately for optional manual review.
    elif weak and context:
        level, rule = "borderline", "borderline_weak_context_far"
        reason = f"weak_context_gap_over_{PROXIMITY_LIMIT}"
    elif weak and (first_person or attitude):
        level, rule = "borderline", "borderline_partial_personal_evidence"
        reason = "only_first_person_or_attitude"
    # E: one unsupported weak keyword is insufficient.
    elif len(weak) == 1:
        level, rule = "irrelevant", "E_single_unsupported_weak_keyword"
        reason = "single_weak_keyword_without_support"
    else:
        level, rule, reason = "irrelevant", "no_project_keyword", "no_project_keyword"

    return pd.Series({
        "relevance_level": level,
        "relevance_rule": rule,
        "matched_strong_terms": json.dumps(strong, ensure_ascii=False),
        "matched_weak_terms": json.dumps(weak, ensure_ascii=False),
        "matched_context_terms": json.dumps(context, ensure_ascii=False),
        "matched_first_person_terms": json.dumps(first_person, ensure_ascii=False),
        "matched_attitude_terms": json.dumps(attitude, ensure_ascii=False),
        "matched_ambiguous_patterns": json.dumps(ambiguous, ensure_ascii=False),
        "weak_context_min_gap": "" if gap is None else gap,
        "relevance_exclusion_reason": reason,
    })

relevance_evidence = cleaned["微博正文"].apply(score_relevance)
relevance_scored = pd.concat([cleaned.reset_index(drop=True), relevance_evidence], axis=1)
relevant = relevance_scored[
    relevance_scored["relevance_level"] == "high_relevance"
].copy()
borderline = relevance_scored[
    relevance_scored["relevance_level"] == "borderline"
].copy()
irrelevant = relevance_scored[
    relevance_scored["relevance_level"] == "irrelevant"
].copy()

RELEVANCE_ALL_FILE = OUTPUT_DIR / "relevance_scored_all.csv"
RELEVANT_FILE = OUTPUT_DIR / "high_relevance_before_han_deduplication.csv"
BORDERLINE_FILE = OUTPUT_DIR / "borderline_relevance.csv"
IRRELEVANT_FILE = OUTPUT_DIR / "removed_as_irrelevant.csv"
relevance_scored.to_csv(RELEVANCE_ALL_FILE, index=False, encoding="utf-8-sig")
relevant.to_csv(RELEVANT_FILE, index=False, encoding="utf-8-sig")
borderline.to_csv(BORDERLINE_FILE, index=False, encoding="utf-8-sig")
irrelevant.to_csv(IRRELEVANT_FILE, index=False, encoding="utf-8-sig")

print("High relevance:", len(relevant))
print("Borderline (saved for review):", len(borderline))
print("Irrelevant:", len(irrelevant))
print(relevance_scored["relevance_rule"].value_counts().rename("rows").to_string())


# Deduplicate high-relevance posts by their complete ordered Han (chinese words) sequence.
def han_only(text):
    return "".join(regex.findall(r"\p{Script=Han}", str(text)))

relevant["han_dedup_key"] = relevant["微博正文"].map(han_only)
relevant["published_dt"] = pd.to_datetime(relevant["发布时间"], errors="coerce")
relevant["_original_order"] = range(len(relevant))

# Valid timestamps come first. Within each duplicate group, the earliest timestamp is retained. Ties are resolved deterministically by source file, source row, and original merged order.
ordered = relevant.sort_values(
    ["han_dedup_key", "published_dt", "source_file", "source_row", "_original_order"],
    ascending=[True, True, True, True, True],
    na_position="last",
    kind="mergesort",
)

kept_mask = ~ordered.duplicated("han_dedup_key", keep="first")
retained = ordered[kept_mask].copy()
duplicate_removed = ordered[~kept_mask].copy()

kept_reference_columns = ["han_dedup_key", "发布时间", "source_file", "source_row"]
if "id" in retained.columns:
    kept_reference_columns.insert(1, "id")
kept_reference = retained[kept_reference_columns].copy()
kept_reference = kept_reference.rename(columns={
    "id": "retained_id",
    "发布时间": "retained_发布时间",
    "source_file": "retained_source_file",
    "source_row": "retained_source_row",
})
duplicate_removed = duplicate_removed.merge(
    kept_reference, on="han_dedup_key", how="left", validate="many_to_one"
)
duplicate_removed["duplicate_removal_reason"] = "identical_complete_han_sequence"

def json_unique(values):
    cleaned_values = sorted({str(value).strip() for value in values if str(value).strip()})
    return json.dumps(cleaned_values, ensure_ascii=False)

metadata = relevant.groupby("han_dedup_key", sort=False).agg(
    duplicate_group_size=("han_dedup_key", "size"),
    all_source_files=("source_file", json_unique),
).reset_index()
if "query_keyword" in relevant.columns:
    keyword_metadata = relevant.groupby("han_dedup_key", sort=False).agg(
        all_query_keywords=("query_keyword", json_unique)
    ).reset_index()
    metadata = metadata.merge(keyword_metadata, on="han_dedup_key", how="left")

stratum_columns = [
    column for column in ["block_start", "block_end", "query_keyword"]
    if column in relevant.columns
]
if stratum_columns:
    relevant["_sampling_stratum"] = relevant[stratum_columns].astype(str).agg("|".join, axis=1)
    stratum_metadata = relevant.groupby("han_dedup_key", sort=False).agg(
        all_sampling_strata=("_sampling_stratum", json_unique)
    ).reset_index()
    metadata = metadata.merge(stratum_metadata, on="han_dedup_key", how="left")

retained = retained.merge(metadata, on="han_dedup_key", how="left", validate="one_to_one")
retained = retained.sort_values(
    ["published_dt", "source_file", "source_row"],
    na_position="last",
    kind="mergesort",
)

internal_columns = [
    "published_dt", "_original_order", "_sampling_stratum",
    "cleaning_removal_reason",
]
retained = retained.drop(
    columns=[column for column in internal_columns if column in retained.columns]
)
duplicate_removed = duplicate_removed.drop(
    columns=[column for column in ["published_dt", "_original_order", "_sampling_stratum"]
             if column in duplicate_removed.columns]
)

FINAL_FILE = OUTPUT_DIR / "eligible_candidates_merged_cleaned_relevant_han_deduplicated.csv"
DUPLICATE_REMOVED_FILE = OUTPUT_DIR / "removed_as_han_sequence_duplicates.csv"
retained.to_csv(FINAL_FILE, index=False, encoding="utf-8-sig")
duplicate_removed.to_csv(DUPLICATE_REMOVED_FILE, index=False, encoding="utf-8-sig")

print("High-relevance rows before deduplication:", len(relevant))
print("Duplicate rows removed:", len(duplicate_removed))
print("Final unique rows:", len(retained))
print("Final CSV:", FINAL_FILE)
print("Duplicate audit:", DUPLICATE_REMOVED_FILE)



# Create two interactive ECharts figures from final query_keyword values.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pyecharts"],
    check=True,
)

try:
    from IPython.display import display
except ImportError:
    def display(_object):
        pass
from pyecharts import options as opts
from pyecharts.charts import Bar, Boxplot
from pyecharts.globals import ThemeType

if "query_keyword" not in retained.columns:
    raise ValueError(
        "The final data do not contain query_keyword, so keyword figures cannot be created."
    )

chart_data = retained.copy()
normalized_keyword = (
    chart_data["query_keyword"]
    .where(chart_data["query_keyword"].notna(), "")
    .astype(str)
    .str.strip()
)
missing_keyword_mask = normalized_keyword.str.casefold().isin(
    {"", "nan", "none", "null", "na", "<na>"}
)
missing_keyword_rows = int(missing_keyword_mask.sum())
chart_data = chart_data.loc[~missing_keyword_mask].copy()
chart_data["query_keyword"] = normalized_keyword.loc[~missing_keyword_mask]
if chart_data.empty:
    raise ValueError("All query_keyword values are missing or NaN-like.")

KEYWORD_ENGLISH = {
    "婚姻": "Marriage",
    "结婚": "Getting married",
    "婚恋": "Marriage and dating",
    "恋爱": "Romantic relationships",
    "对象": "Romantic partner",
    "伴侣": "Partner or companion",
    "相亲": "Matchmaking",
    "单身": "Singlehood",
    "不婚": "Non-marriage",
    "晚婚": "Late marriage",
    "恐婚": "Fear of marriage",
    "催婚": "Marriage pressure",
    "婚姻登记": "Marriage registration",
    "彩礼": "Bride price",
    "离婚冷静期": "Divorce cooling-off period",
}
unmapped_keywords = sorted(
    set(chart_data["query_keyword"]) - set(KEYWORD_ENGLISH)
)
if unmapped_keywords:
    raise ValueError(
        "Add English translations for these unexpected query_keyword values: "
        + repr(unmapped_keywords)
    )
chart_data["query_keyword_english"] = chart_data["query_keyword"].map(
    KEYWORD_ENGLISH
)

palette = [
    "#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#F8B195",
    "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#457B9D",
    "#1D3557", "#7A5195", "#EF5675", "#FFA600", "#4D908E",
]

# Figure 1: ranked counts in the final analytic corpus.
keyword_counts = chart_data["query_keyword_english"].value_counts().sort_values()
keyword_shares = (keyword_counts / keyword_counts.sum() * 100).round(2)
keyword_summary = pd.DataFrame({
    "query_keyword_english": keyword_counts.index,
    "retained_posts": keyword_counts.values,
    "share_percent": keyword_shares.values,
}).sort_values("retained_posts", ascending=False)
english_to_chinese = {english: chinese for chinese, english in KEYWORD_ENGLISH.items()}
keyword_summary.insert(
    0,
    "query_keyword_chinese",
    keyword_summary["query_keyword_english"].map(english_to_chinese),
)
KEYWORD_COUNTS_FILE = OUTPUT_DIR / "query_keyword_final_counts.csv"
keyword_summary.to_csv(KEYWORD_COUNTS_FILE, index=False, encoding="utf-8-sig")

count_items = [
    opts.BarItem(
        name=keyword,
        value=int(value),
        itemstyle_opts=opts.ItemStyleOpts(
            color=palette[index % len(palette)],
            border_radius=[0, 7, 7, 0],
        ),
    )
    for index, (keyword, value) in enumerate(keyword_counts.items())
]
count_chart = (
    Bar(init_opts=opts.InitOpts(
        theme=ThemeType.LIGHT, width="1200px", height="720px",
        bg_color="#FAFAF8",
    ))
    .add_xaxis(keyword_counts.index.tolist())
    .add_yaxis(
        "Retained posts", count_items, category_gap="38%",
        label_opts=opts.LabelOpts(is_show=True, position="right", font_size=12),
    )
    .reversal_axis()
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Marriage-related Weibo posts retained by query keyword",
            subtitle=(
                "Final cleaned, high-relevance, deduplicated corpus; "
                "counts describe this sample, not total Weibo popularity."
            ),
            pos_left="center",
        ),
        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                    type_="png", title="Save as PNG", pixel_ratio=2
                ),
                data_view=opts.ToolBoxFeatureDataViewOpts(
                    title="View data", is_read_only=True
                ),
            ),
        ),
        xaxis_opts=opts.AxisOpts(
            name="Number of retained posts",
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        yaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(font_size=13),
        ),
        graphic_opts=[opts.GraphicText(
            graphic_item=opts.GraphicItem(left="center", bottom=8),
            graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                text=(
                    f"N = {len(chart_data):,} posts with valid query_keyword; "
                    f"{missing_keyword_rows:,} missing/NaN-like rows excluded."
                ),
                font="12px sans-serif", graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                    fill="#666666"
                ),
            ),
        )],
    )
)
COUNT_CHART_FILE = OUTPUT_DIR / "figure1_query_keyword_counts.html"
count_chart.render(str(COUNT_CHART_FILE))

# Figure 2: distribution of post length by keyword. Length is measured using Han (chinese) characters only, so URLs, Latin text, punctuation, spaces, and emoji do not inflate the comparison.
chart_data["han_character_count"] = chart_data["微博正文"].map(
    lambda text: len(regex.findall(r"\p{Script=Han}", str(text)))
)
keyword_order = (
    chart_data.groupby("query_keyword_english")["han_character_count"]
    .median()
    .sort_values()
    .index.tolist()
)
length_distributions = [
    chart_data.loc[
        chart_data["query_keyword_english"] == keyword,
        "han_character_count",
    ].tolist()
    for keyword in keyword_order
]

tukey_box_data = []
full_range_box_data = []
tukey_statistics = {}
for keyword, values in zip(keyword_order, length_distributions):
    values_series = pd.Series(values, dtype="float64")
    q1 = float(values_series.quantile(0.25))
    median = float(values_series.median())
    q3 = float(values_series.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower_whisker = float(values_series[values_series >= lower_fence].min())
    upper_whisker = float(values_series[values_series <= upper_fence].max())
    outlier_count = int(
        ((values_series < lower_fence) | (values_series > upper_fence)).sum()
    )
    tukey_box_data.append([
        lower_whisker, q1, median, q3, upper_whisker
    ])
    full_range_box_data.append([
        float(values_series.min()), q1, median, q3, float(values_series.max())
    ])
    tukey_statistics[keyword] = {
        "tukey_lower_whisker": lower_whisker,
        "tukey_upper_whisker": upper_whisker,
        "tukey_outlier_posts": outlier_count,
        "tukey_outlier_percent": outlier_count / len(values_series) * 100,
    }

length_summary = (
    chart_data.groupby(["query_keyword", "query_keyword_english"])
    ["han_character_count"]
    .agg(
        posts="size",
        mean_han_characters="mean",
        median_han_characters="median",
        minimum_han_characters="min",
        q1_han_characters=lambda values: values.quantile(0.25),
        q3_han_characters=lambda values: values.quantile(0.75),
        maximum_han_characters="max",
    )
    .reset_index()
    .sort_values("median_han_characters", ascending=False)
)
for column in [
    "mean_han_characters", "median_han_characters",
    "q1_han_characters", "q3_han_characters",
]:
    length_summary[column] = length_summary[column].round(2)
for statistic in [
    "tukey_lower_whisker", "tukey_upper_whisker",
    "tukey_outlier_posts", "tukey_outlier_percent",
]:
    length_summary[statistic] = length_summary["query_keyword_english"].map(
        lambda keyword: tukey_statistics[keyword][statistic]
    )
length_summary["tukey_outlier_percent"] = (
    length_summary["tukey_outlier_percent"].round(2)
)
LENGTH_SUMMARY_FILE = OUTPUT_DIR / "query_keyword_text_length_summary.csv"
length_summary.to_csv(LENGTH_SUMMARY_FILE, index=False, encoding="utf-8-sig")

length_chart = Boxplot(init_opts=opts.InitOpts(
    theme=ThemeType.LIGHT, width="1400px", height="760px",
    bg_color="#FAFAF8",
))
length_chart.add_xaxis(keyword_order)
length_chart.add_yaxis(
    "Post length",
    tukey_box_data,
    itemstyle_opts=opts.ItemStyleOpts(
        color="#6C5B7B", border_color="#355C7D", border_width=1.5
    ),
)
length_chart.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Typical distribution of Weibo post length by query keyword",
        subtitle=(
            "Standard Tukey box plot: whiskers stop at the most extreme values "
            "within 1.5 × IQR, so a few very long posts do not compress the boxes."
        ),
        pos_left="center",
    ),
    legend_opts=opts.LegendOpts(is_show=False),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
    toolbox_opts=opts.ToolboxOpts(
        is_show=True,
        feature=opts.ToolBoxFeatureOpts(
            save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                type_="png", title="Save as PNG", pixel_ratio=2
            ),
            data_view=opts.ToolBoxFeatureDataViewOpts(
                title="View data", is_read_only=True
            ),
        ),
    ),
    xaxis_opts=opts.AxisOpts(
        name="Query keyword",
        axislabel_opts=opts.LabelOpts(rotate=28, interval=0, font_size=11),
    ),
    yaxis_opts=opts.AxisOpts(
        name="Chinese characters per post",
        min_=0,
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
    graphic_opts=[opts.GraphicText(
        graphic_item=opts.GraphicItem(left="center", bottom=5),
        graphic_textstyle_opts=opts.GraphicTextStyleOpts(
            text=(
                "Extreme posts remain in the dataset and summary CSV but are omitted "
                "from this main view; see Figure 2B for the complete range."
            ),
            font="12px sans-serif", graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                fill="#666666"
            ),
        ),
    )],
)
LENGTH_CHART_FILE = OUTPUT_DIR / "figure2a_post_length_tukey_boxplot.html"
length_chart.render(str(LENGTH_CHART_FILE))

# Figure 2B preserves each keyword's actual minimum and maximum while a logarithmic y-axis compresses the long right tail. This is a transparency check rather than the preferred main report figure.
full_range_chart = Boxplot(init_opts=opts.InitOpts(
    theme=ThemeType.LIGHT, width="1400px", height="760px",
    bg_color="#FAFAF8",
))
full_range_chart.add_xaxis(keyword_order)
full_range_chart.add_yaxis(
    "Post length",
    full_range_box_data,
    itemstyle_opts=opts.ItemStyleOpts(
        color="#C06C84", border_color="#6C5B7B", border_width=1.5
    ),
)
full_range_chart.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Full range of Weibo post length by query keyword",
        subtitle=(
            "Logarithmic y-axis; every keyword's observed minimum and maximum are "
            "included, allowing rare very long posts to remain visible."
        ),
        pos_left="center",
    ),
    legend_opts=opts.LegendOpts(is_show=False),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
    toolbox_opts=opts.ToolboxOpts(
        is_show=True,
        feature=opts.ToolBoxFeatureOpts(
            save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                type_="png", title="Save as PNG", pixel_ratio=2
            ),
            data_view=opts.ToolBoxFeatureDataViewOpts(
                title="View data", is_read_only=True
            ),
        ),
    ),
    xaxis_opts=opts.AxisOpts(
        name="Query keyword",
        axislabel_opts=opts.LabelOpts(rotate=28, interval=0, font_size=11),
    ),
    yaxis_opts=opts.AxisOpts(
        type_="log", log_base=10, min_=1,
        name="Chinese characters per post (log scale)",
        splitline_opts=opts.SplitLineOpts(is_show=True),
    ),
    graphic_opts=[opts.GraphicText(
        graphic_item=opts.GraphicItem(left="center", bottom=5),
        graphic_textstyle_opts=opts.GraphicTextStyleOpts(
            text=(
                "Use Figure 2A for comparison of typical posts; this log-scale panel "
                "shows the long tail without deleting or winsorizing observations."
            ),
            font="12px sans-serif", graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                fill="#666666"
            ),
        ),
    )],
)
FULL_RANGE_CHART_FILE = OUTPUT_DIR / "figure2b_post_length_full_range_log_scale.html"
full_range_chart.render(str(FULL_RANGE_CHART_FILE))

print("Figure 1:", COUNT_CHART_FILE)
display(count_chart.render_notebook())
print("Figure 2A (recommended main figure):", LENGTH_CHART_FILE)
display(length_chart.render_notebook())
print("Figure 2B (full-range robustness view):", FULL_RANGE_CHART_FILE)
display(full_range_chart.render_notebook())
print("Use the camera icon in the upper-right toolbox to save any chart as PNG.")
