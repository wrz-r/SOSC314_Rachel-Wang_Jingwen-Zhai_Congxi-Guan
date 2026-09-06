# Import Path so that we can define and check the folder where the repository will be stored.
from pathlib import Path
# Import subprocess so that we can run Git commands from Python.
import subprocess

# Define the folder where the full-year Weibo Search repository will be stored in Google Colab.
FULL_REPO = Path("/content/weibo-search-2025-full")
# Check whether the repository has already been downloaded.
# If the folder does not exist, clone the repository from GitHub.
if not FULL_REPO.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--depth", "1",
            "https://github.com/dataabc/weibo-search.git",
            str(FULL_REPO)
        ],
        check=True
    )
# Print the local repository path so that we can confirm where the files are stored.
print("Full-year repository:", FULL_REPO)

# Import getpass so that we can enter the Weibo cookie without displaying
from getpass import getpass
# Ask me to enter their Weibo cookie.
# The cookie is stored in WEIBO_COOKIE, but the entered value remains hidden to protect sensitive account information.
WEIBO_COOKIE = getpass("Paste your Weibo cookie here: ")
# Confirm that the cookie has been stored without revealing its value.
if WEIBO_COOKIE:
    print("Cookie loaded successfully. Its value is hidden.")
else:
    print("No cookie was entered.")


# Install the Python packages required by the Weibo Search repository.
!pip -q install -r /content/weibo-search-2025-full/requirements.txt
# Reinstall the specified versions of Scrapy and Twisted to ensure that they are compatible with the crawler and to avoid errors caused by other installed versions.
!pip -q install --force-reinstall "Scrapy==2.12.0" "Twisted==24.11.0"
# Display the installed Scrapy version and the versions of its main dependencies.
!scrapy version -v

# Define 2025 as the year used for the pilot data collection.
PILOT_YEAR = 2025
# Define the Chinese keywords that will be used to search for Weibo posts related to marriage and fertility.
KEYWORDS = ["婚姻","结婚","晚婚","不婚","生育","生孩子","不婚不育"]

# Set the maximum number of Weibo posts collected for each keyword in each month.
MAX_RESULTS_PER_KEYWORD_MONTH = 30

# Wait 30 seconds between consecutive page requests.
DOWNLOAD_DELAY = 30
# Calculate the theoretical maximum number of observations that could be collected.
print("Maximum possible observations:",len(KEYWORDS) * 12 * MAX_RESULTS_PER_KEYWORD_MONTH)


# Import pandas so that we can create and organize the monthly date ranges.
import pandas as pd
# Create a sequence containing the first day of every month from January 2025 to January 2026. The January 2026 date is included only to calculate the end date of December 2025.
month_starts = pd.date_range(
    start="2025-01-01",
    end="2026-01-01",
    freq="MS"
)
# Create an empty list to store the start date and end date of each month.
month_ranges = []
# Loop through the 12 months of 2025.
for i in range(len(month_starts) - 1):
    # Select the first day of the current month.
    start_date = month_starts[i]
    # Select the first day of the following month.
    next_month = month_starts[i + 1]
    # Calculate the final day of the current month by subtracting one day from the first day of the following month.
    end_date = next_month - pd.Timedelta(days=1)
    # Store the month label, start date, and end date as strings.
    month_ranges.append({
        "month": start_date.strftime("%Y-%m"),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    })
# Convert the list of monthly date ranges into a pandas DataFrame.
month_ranges = pd.DataFrame(month_ranges)

display(month_ranges)


# Run the pilot data collection across all months of the selected year.
# Each collection task corresponds to one keyword in one month.
# Because the number of results is capped, this is a full-year pilot collection rather than a complete collection of all relevant Weibo posts.

# Import os so that we can create a copy of the current system environment and pass crawler settings through environment variables.
import os
# Import json so that Python lists containing Chinese text can be converted into JSON strings that the crawler can read.
import json
# Import time so that we can measure how long each collection task takes.
import time

# Define the file used to record the progress and outcome of each collection task.
LOG_FILE = Path("/content/weibo_2025_collection_log.csv")

# Check whether a collection log from an earlier run already exists.
# If it exists, read the previous records so that completed tasks can be skipped.
if LOG_FILE.exists():
    collection_log = pd.read_csv(LOG_FILE)
else:
    collection_log = pd.DataFrame()

# Create an empty list to store records generated during the current run.
run_records = []

# If the previous collection log contains records, construct a set containing all previously recorded month-keyword combinations.
if not collection_log.empty:
    completed_tasks = set(
        zip(
            collection_log["month"],
            collection_log["keyword"]
        )
    )
else:
    completed_tasks = set()

# Iterate through every monthly date range in the month_ranges DataFrame.
for _, month_row in month_ranges.iterrows():
    # Extract the month label and the corresponding start and end dates.
    month = month_row["month"]
    start_date = month_row["start_date"]
    end_date = month_row["end_date"]
    
    # Run a separate collection task for each search keyword.
    for keyword in KEYWORDS:

        task = (month, keyword)

        # Skip the current task if the same month-keyword combination has already been recorded in the collection log.
        if task in completed_tasks:
            print(
                f"Already completed: {month} | {keyword}"
            )
            continue

        # Define the CSV file in which the crawler stores results for the current keyword.
        output_file = (
            FULL_REPO
            / "结果文件"
            / keyword
            / f"{keyword}.csv"
        )

        # Count how many rows already exist in the keyword's output file before running the current monthly collection task.
        if output_file.exists():
            rows_before = len(
                pd.read_csv(output_file)
            )
        else:
            rows_before = 0
        
        
        # Display the current month, keyword, and date range so that we can monitor the collection progress.
        print(
            f"\nCollecting: {month} | {keyword}"
        )
        print(
            f"Date range: {start_date} to {end_date}"
        )
        # Create a copy of the current system environment.
        environment = os.environ.copy()
        # Pass the Weibo login cookie to the crawler.
        environment["WEIBO_COOKIE"] = WEIBO_COOKIE
        # Pass only the current keyword to the crawler.
        environment["WEIBO_KEYWORDS"] = json.dumps(
            [keyword],
            ensure_ascii=False
        )
        # Pass the beginning and ending dates of the current month to define the collection period.
        environment["WEIBO_START_DATE"] = start_date
        environment["WEIBO_END_DATE"] = end_date

        # Restrict the search to original Weibo posts.
        environment["WEIBO_TYPE"] = "1"
        # Apply no additional content-format restriction.
        environment["WEIBO_CONTAIN_TYPE"] = "0"
        # Apply no geographical restriction to the search results.
        environment["WEIBO_REGION"] = json.dumps(
            ["全部"],
            ensure_ascii=False
        )

        environment["WEIBO_LIMIT_RESULT"] = str(
            MAX_RESULTS_PER_KEYWORD_MONTH
        )
        # Disable the collection of IP-location information.
        environment["WEIBO_FETCH_IP"] = "0"

        start_time = time.time()

        
        # Run the Scrapy search spider using the settings defined above.
        result = subprocess.run(
            [
                "scrapy",
                "crawl",
                "search",
                "-s", "LOG_LEVEL=ERROR",
                "-s", f"DOWNLOAD_DELAY={DOWNLOAD_DELAY}",
                "-s", "RANDOMIZE_DOWNLOAD_DELAY=True"
            ],
            # Run the Scrapy command from the repository's root folder.
            cwd=FULL_REPO,
            # Pass the customized environment variables to the crawler.
            env=environment,
            # Store the crawler's standard output and error messages instead of printing all of them directly in the notebook.
            capture_output=True,
            text=True
        )

        # Count the number of rows in the output file after the crawler finishes.
        if output_file.exists():
            rows_after = len(
                pd.read_csv(output_file)
            )
        else:
            rows_after = 0
            
        # Create a record describing the current collection task and its outcome.
        current_record = {
            "month": month,
            "keyword": keyword,
            "start_date": start_date,
            "end_date": end_date,
            "new_rows": new_rows,
            "return_code": result.returncode,
            "elapsed_seconds": elapsed_seconds
        }
        
        
        # Add the current task record to the list of records generated during this notebook run.
        run_records.append(current_record)

        # preserve
        updated_log = pd.concat(
            [collection_log, pd.DataFrame(run_records)],
            ignore_index=True
        )

        updated_log.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
        
        # Print a short summary of the completed task.
        print(
            f"Finished: {month} | {keyword}"
        )
        print("New rows:", new_rows)
        print("Time:", elapsed_seconds, "seconds")
        print("Return code:", result.returncode)
        
        # A return code other than 0 indicates that the Scrapy process encountered an error.
        if result.returncode != 0:
            print("Last error messages:")
            print(result.stderr[-1500:])


# Review all the results and combine
# Reload the latest collection log from the saved CSV file.
collection_log = pd.read_csv(LOG_FILE)
display(collection_log)

# Create an empty list to store the DataFrame collected for each search keyword.
all_2025_data = []
# Loop through all marriage- and fertility-related search keywords.
for keyword in KEYWORDS:
    # Define the expected location of the CSV output file
    file_path = (
        FULL_REPO
        / "结果文件"
        / keyword
        / f"{keyword}.csv"
    )
    # Continue only if an output file exists for the current keyword.
    if file_path.exists():
        # Read the keyword-specific CSV file into a pandas DataFrame.
        keyword_data = pd.read_csv(
            file_path,
            dtype={
                "id": str,
                "user_id": str,
                "retweet_id": str
            }
        )
        # Add the search keyword used to retrieve each record.
        keyword_data["search_query"] = keyword
        # Add the current keyword's DataFrame to the list of available datasets.
        all_2025_data.append(
            keyword_data
        )
# Check whether at least one keyword-specific dataset was found.
if all_2025_data:
    # Combine all available keyword-specific DataFrames into one dataset.
    raw_posts_2025 = pd.concat(
        all_2025_data,
        ignore_index=True
    )
    # Print the total number of raw records across all search-query files.
    print(
        "Total raw records collected:",
        len(raw_posts_2025)
    )
    # Display the first five rows so that we can inspect the structure and content of the combined dataset.
    display(raw_posts_2025.head())
else:
    print("No 2025 data found.")


# Define the location and filename for the combined raw dataset.
RAW_OUTPUT = Path("/content/weibo_marriage_fertility_2025_raw.csv")
# Export the combined raw Weibo dataset as a CSV file.
raw_posts_2025.to_csv(
    RAW_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)
# Print the output path so that we can confirm where the file was saved.
print("Saved to:", RAW_OUTPUT)



# Import the regular-expression module so that we can identify text patterns during the cleaning process.
import re

# Define the location of the original raw dataset.
# This file will only be read and will not be overwritten during cleaning.
RAW_FILE = Path("/content/weibo_marriage_fertility_2025_raw.csv")

# Define the location where the cleaned dataset will be saved.
CLEANED_FILE = Path("/content/weibo_marriage_fertility_2025_cleaned.csv")

# Define the location where removed observations will be saved.
REMOVED_FILE = Path("/content/weibo_marriage_fertility_2025_removed_rows.csv")

# Read the original raw dataset.
df = pd.read_csv(
    RAW_FILE,
    dtype={
        "id": str,
        "user_id": str,
        "retweet_id": str
    }
)
# Verify that the dataset contains the text variable required for the subsequent cleaning process.
if "微博正文" not in df.columns:
    raise KeyError(
        f"找不到“微博正文”列。当前列名为：{df.columns.tolist()}"
    )
# Create a separate text Series from the Weibo-content column for use in the subsequent text-based cleaning rules.
text = (
    df["微博正文"]
    .fillna("")
    .astype(str)
)
# Report the number of observations in the raw dataset before any cleaning rules are applied.
print("清理前总行数:", len(df))


# Define a list of words and symbols associated with content that is considered irrelevant to the research topic based on manual inspection.
keywords_to_remove = ["小说", "京东", "拼多多", "超话", "笔趣阁", "国漫", "』"]
# Combine the exclusion terms into a single regular-expression pattern.
keyword_pattern = "|".join(
    re.escape(keyword)
    for keyword in keywords_to_remove
)
# Create a Boolean indicator identifying posts that contain at least one term or symbol from the exclusion list.
contains_excluded_keyword = text.str.contains(
    keyword_pattern,
    regex=True,
    na=False
)

# Define a regular-expression pattern covering the main Unicode ranges used for Japanese characters.
japanese_pattern = (
    r"[\u3040-\u309F"
    r"\u30A0-\u30FF"
    r"\u31F0-\u31FF"
    r"\uFF66-\uFF9D]"
)
# Create a Boolean indicator identifying posts that contain at least one Japanese kana character from the Unicode ranges defined above.
contains_japanese = text.str.contains(
    japanese_pattern,
    regex=True,
    na=False
)

# Create an empty list to store the number of rows matched by each individual removal rule.
removal_counts = []
# Calculate how many rows contain each term or symbol in the predefined exclusion list.
for keyword in keywords_to_remove:
    # Create a Boolean indicator for the current exclusion term and count the number of matching rows.
    keyword_count = text.str.contains(
        re.escape(keyword),
        regex=True,
        na=False
    ).sum()

    removal_counts.append({
        "removal_reason": keyword,
        "matched_rows": int(keyword_count)
    })
# Add the number of rows containing at least one Japanese character to the same summary.
removal_counts.append({
    "removal_reason": "Japanese hiragana/katakana",
    "matched_rows": int(contains_japanese.sum())
})

removal_summary = pd.DataFrame(removal_counts)
# Display the number of rows matched by each removal rule.
display(removal_summary)

# Combine the keyword-based and Japanese-character removal conditions.
rows_to_delete = (
    contains_excluded_keyword | contains_japanese
)
# Count and report the number of unique rows that satisfy at least one removal condition.
print(
    "符合任意删除条件的总行数:", int(rows_to_delete.sum())
)


# Select all rows that do not satisfy any removal condition.
df_cleaned = df.loc[~rows_to_delete].copy()
# Save the retained observations as the cleaned dataset.
df_cleaned.to_csv(
    CLEANED_FILE,
    index=False,
    encoding="utf-8-sig"
)
# Report the number of rows before cleaning and the number retained after cleaning.
print("清理前总行数:", len(df))
print("清理后总行数:", len(df_cleaned))
# Display the location of the cleaned dataset.
print("\n清理后文件保存在:")
print(CLEANED_FILE)












