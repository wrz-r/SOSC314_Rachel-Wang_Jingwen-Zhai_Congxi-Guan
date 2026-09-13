#!/usr/bin/env python3
"""Collect marriage/fertility articles from official Chinese media websites.

Searches official sites by keyword, downloads full article text, and exports
one article per CSV row. SQLite checkpoints make interrupted runs resumable.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing packages. Run: /usr/local/bin/python3 -m pip install "
        "requests beautifulsoup4"
    ) from error


KEYWORD_GROUPS = {
    "marriage": [
        "婚姻", "结婚", "婚恋", "恋爱", "对象", "伴侣", "相亲", "单身",
        "不婚", "晚婚", "恐婚", "催婚", "婚姻登记", "彩礼", "离婚冷静期",
    ]

}
QUERY_KEYWORDS = sorted({word for words in KEYWORD_GROUPS.values() for word in words})

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
OUTPUT_FIELDS = [
    "media", "date", "title", "text", "text_length", "url",
    "search_keywords", "matched_groups", "matched_keywords",
]
CANDIDATE_FIELDS = [
    "media", "search_keyword", "search_title", "search_date", "url",
    "search_content", "study_year",
]

# GDELT only discovers URLs for sites without any usable search endpoint.
# Article text is always downloaded from the official domain itself.
GDELT_SOURCES = {
    "Xinhua": ("news.cn", ("news.cn", "xinhuanet.com")),
    "Economic Daily / CE.cn": ("ce.cn", ("ce.cn",)),
}


def clean_text(value: object) -> str:
    """Remove HTML tags and repeated whitespace."""
    if value is None:
        return ""
    raw = str(value)
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True) if "<" in raw else raw
    return re.sub(r"\s+", " ", text).strip()


def parse_date(value: object) -> datetime | None:
    """Parse timestamps, metadata dates, and dates embedded in URLs."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        stamp = float(value) / 1000 if value > 10**11 else float(value)
        try:
            return datetime.fromtimestamp(stamp)
        except (ValueError, OSError):
            return None
    text = str(value).strip().replace("年", "-").replace("月", "-").replace("日", "")
    match = re.search(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if not match:
        match = re.search(r"(20\d{2})(\d{2})(\d{2})", text)
    if match:
        try:
            return datetime(*map(int, match.groups()))
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def nearby_date(element) -> datetime | None:
    """Look for a publication date near a search-result link."""
    current = element
    for _ in range(6):
        if current is None:
            return None
        date = parse_date(current.get_text(" ", strip=True))
        if date:
            return date
        current = current.parent
    return None


def matches(text: str) -> tuple[list[str], list[str]]:
    """Return topic groups and study keywords found in an article."""
    found = {
        group: [word for word in words if word in text]
        for group, words in KEYWORD_GROUPS.items()
    }
    found = {group: words for group, words in found.items() if words}
    words = sorted({word for group_words in found.values() for word in group_words})
    return sorted(found), words


def is_mainly_chinese(text: str) -> bool:
    """Require substantial, predominantly Chinese text for China Daily rows."""
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return chinese >= 200 and chinese / max(chinese + latin, 1) >= 0.60


def make_candidate(media, keyword, title, date, url, content, year) -> dict:
    return {
        "media": media,
        "search_keyword": keyword,
        "search_title": clean_text(title),
        "search_date": date.strftime("%Y-%m-%d") if isinstance(date, datetime) else (date or ""),
        "url": url,
        "search_content": clean_text(content),
        "study_year": year,
    }


def year_bounds_ms(year: int) -> tuple[int, int]:
    """Return the first and last millisecond of a study year."""
    start = int(datetime(year, 1, 1).timestamp()) * 1000
    end = int(datetime(year, 12, 31, 23, 59, 59).timestamp()) * 1000
    return start, end


def search_people(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search the People's Daily website, bounded to the study year.

    The HTTPS host answers with a 301 to plain HTTP. Following that redirect
    rewrites the POST into an empty GET that still returns valid JSON with no
    records, which looks like "no matches" rather than a failure. Redirects are
    therefore disabled, the HTTP endpoint is preferred, and the millisecond
    startTime/endTime fields pin the results to the study year.
    """
    endpoints = [
        "http://search.people.cn/search-platform/front/search",
        "https://search.people.cn/search-platform/front/search",
        "http://search.people.cn/api-search/front/search",
    ]
    start_ms, end_ms = year_bounds_ms(year)
    output, working_endpoint = [], None
    for page in range(1, pages + 1):
        payload = {
            "key": keyword, "page": page, "limit": 10, "sortType": 2,
            "type": 0, "hasTitle": True, "hasContent": True,
            "isFuzzy": True, "startTime": start_ms, "endTime": end_ms,
        }
        data = None
        for endpoint in ([working_endpoint] if working_endpoint else endpoints):
            try:
                response = session.post(
                    endpoint, json=payload, timeout=25, allow_redirects=False
                )
                if response.status_code != 200:
                    continue
                possible = response.json()
                if isinstance(possible.get("data"), dict):
                    data, working_endpoint = possible, endpoint
                    break
            except (requests.RequestException, ValueError):
                continue
        if data is None:
            raise requests.RequestException("All People.cn search endpoints failed")
        records = (data.get("data") or {}).get("records") or []
        if not records:
            break
        for item in records:
            date = parse_date(item.get("displayTime"))
            if date and date.year == year and item.get("url"):
                output.append(make_candidate(
                    "People.cn", keyword, item.get("title"), date,
                    item["url"], item.get("content"), year,
                ))
    return output


def search_cctv(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search CCTV News, bounded to the study year.

    The backend's vtime field is the video-duration filter and datepid only
    offers relative presets, so neither can pin results to a past year. Web
    results are instead paged in descending publication order (sort=date) and
    the sweep stops as soon as a page's oldest story predates the study year,
    which reaches that year without walking the entire archive.
    """
    output = []
    year_start = datetime(year, 1, 1)
    page, cap = 1, max(pages, 60)
    while page <= cap:
        response = session.get(
            "https://search.cctv.com/search.php",
            params={
                "qtext": keyword, "page": page, "type": "web", "sort": "date",
                "datepid": 1, "vtime": -1, "is_search": 1,
            },
            timeout=25,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a", id=lambda value: value and str(value).startswith("web_content_"))
        if not links:
            break
        page_dates = []
        for link in links:
            href = link.get("href", "")
            target = parse_qs(urlparse(href).query).get("targetpage", [None])[0]
            url = unquote(target) if target else href
            date = nearby_date(link) or parse_date(url)
            if not date:
                continue
            page_dates.append(date)
            host = urlparse(url).netloc.lower()
            if host.endswith("cctv.com") and date.year == year:
                output.append(make_candidate("CCTV", keyword, link.get_text(), date, url, "", year))
        if page_dates and min(page_dates) < year_start:
            break
        page += 1
        time.sleep(0.3)
    return output


def year_days(year: int):
    """Yield every calendar day of a study year."""
    day = datetime(year, 1, 1)
    while day.year == year:
        yield day
        day += timedelta(days=1)


def search_chinanews_day(session, day: datetime, keywords: list[str]) -> list[dict]:
    """Read China News Service's daily archive page.

    chinanews.com.cn publishes one static page per day listing every story
    released that day (700-800 links) with the headline as the link text. Its
    own site search ignores the submitted date range, so the date-addressed
    archive is walked instead. Headlines pre-filter the links and the full text
    is verified after download.
    """
    page_url = f"https://www.chinanews.com.cn/scroll-news/{day:%Y}/{day:%m%d}/news.shtml"
    response = session.get(page_url, timeout=30)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    pattern = re.compile(
        rf"^https?://[^/]*chinanews\.com\.cn/[a-z]+/{day:%Y}/{day:%m-%d}/\d+\.shtml$"
    )
    rows = []
    for link in soup.find_all("a", href=True):
        url = urljoin(page_url, link["href"])
        if not pattern.match(url):
            continue
        title = clean_text(link.get_text())
        hits = sorted(word for word in keywords if title and word in title)
        if hits:
            rows.append(make_candidate(
                "China News Service", "|".join(hits), title, day, url, "", day.year,
            ))
    return rows


# The digital editions of Guangming Daily and China Youth Daily publish a
# per-issue plate index (版面) whose pages list every article of that issue with
# the headline as the link text. The news sites behind them reject scripted
# access (403 or a hijacked search domain), so the printed edition - the
# canonical text of both outlets - is the date-addressable corpus.
EPAPERS = {
    "China Youth Daily / youth.cn": ("http://zqb.cyol.com/html", "zgqnb"),
    "Guangming Daily / GMW.cn": ("http://epaper.gmw.cn/gmrb/html", "gmrb"),
}
EPAPER_MEDIA = set(EPAPERS)


def search_epaper_day(session, media: str, day: datetime, keywords: list[str]) -> list[dict]:
    """Collect one issue of a newspaper's digital edition, filtered by headline."""
    base, token = EPAPERS[media]
    issue_dir = f"{base}/{day:%Y-%m}/{day:%d}"
    plate_pattern = re.compile(rf"nbs\.D110000{token}_(\d{{2}})\.htm$")
    article_marker = f"nw.D110000{token}_"
    first_url = f"{issue_dir}/nbs.D110000{token}_01.htm"
    response = session.get(first_url, timeout=30)
    if response.status_code != 200:
        return []
    first_page = BeautifulSoup(response.content, "html.parser")
    plates = sorted({
        int(match.group(1))
        for link in first_page.find_all("a", href=True)
        if (match := plate_pattern.search(link["href"]))
    }) or [1]
    rows, seen = [], set()
    for plate in plates:
        page_url = f"{issue_dir}/nbs.D110000{token}_{plate:02d}.htm"
        if plate == 1:
            soup = first_page
        else:
            try:
                plate_response = session.get(page_url, timeout=30)
                if plate_response.status_code != 200:
                    continue
                soup = BeautifulSoup(plate_response.content, "html.parser")
            except requests.RequestException:
                continue
        for link in soup.find_all("a", href=True):
            if article_marker not in link["href"]:
                continue
            title = clean_text(link.get_text())
            url = urljoin(page_url, link["href"])
            hits = sorted(word for word in keywords if title and word in title)
            if hits and url not in seen:
                seen.add(url)
                rows.append(make_candidate(media, "|".join(hits), title, day, url, "", day.year))
    return rows


def search_youth_day(session, day: datetime, keywords: list[str]) -> list[dict]:
    """China Youth Daily digital edition."""
    return search_epaper_day(session, "China Youth Daily / youth.cn", day, keywords)


def search_gmw_day(session, day: datetime, keywords: list[str]) -> list[dict]:
    """Guangming Daily digital edition."""
    return search_epaper_day(session, "Guangming Daily / GMW.cn", day, keywords)


def search_chinadaily(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search Chinese-language China Daily through its JSON API.

    The API accepts no date filter. It can sort by publication time, so we walk
    it newest-first and stop once a page's oldest story predates the study year.
    Walking down from the present only crosses the handful of years after the
    target; walking up from the archive start has to cross every earlier year,
    which is thousands of records for a common word and reliably times out.
    --max-pages is ignored here because the page count is data-driven; a hard
    safety cap applies.
    """
    search_page = "https://newssearch.chinadaily.com.cn/cn/search"
    landing = session.get(search_page, params={"query": keyword}, timeout=25)
    landing.raise_for_status()
    output = []
    ajax_headers = {
        # requests has already percent-encoded the Chinese keyword in this URL.
        "Referer": landing.url,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    year_start = datetime(year, 1, 1)
    page, total_pages = 0, None
    while page < 400:
        response = session.get(
            "https://newssearch.chinadaily.com.cn/rest/cn/search",
            params={
                "keywords": keyword, "sort": "dp", "page": page,
                "curType": "story", "size": 100,
            },
            headers=ajax_headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("content") or []
        if not records:
            break
        total_pages = payload.get("totalPages") or total_pages
        dated = [
            (parse_date(item.get("publishTime")) or parse_date(item.get("url")), item)
            for item in records
        ]
        for date, item in dated:
            url = item.get("url", "")
            host = urlparse(url).netloc.lower()
            if date and date.year == year and host.endswith("chinadaily.com.cn"):
                output.append(make_candidate(
                    "China Daily", keyword, item.get("title"), date, url,
                    item.get("plainText") or item.get("content"), year,
                ))
        oldest = min((date for date, _ in dated if date), default=None)
        page += 1
        if oldest and oldest < year_start:
            break
        if total_pages and page >= total_pages:
            break
        time.sleep(0.3)
    return output


def search_workercn(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search Workers' Daily / Workercn through its JSON endpoint."""
    output = []
    for page in range(pages):
        response = session.get(
            "https://www.workercn.cn/cms/front/search/result",
            params={
                "query": keyword, "siteID": 122, "pageIndex": page,
                "pageSize": 15, "sort": "publishDate",
                "startDate": f"{year}-01-01", "endDate": f"{year}-12-31",
            },
            timeout=30,
        )
        response.raise_for_status()
        records = (response.json().get("data") or {}).get("data") or []
        if not records:
            break
        for item in records:
            url = item.get("artUrl") or item.get("url") or ""
            date = parse_date(item.get("publishDate")) or parse_date(url)
            if urlparse(url).netloc.lower().endswith("workercn.cn") and date and date.year == year:
                output.append(make_candidate(
                    "Workers' Daily / Workercn", keyword, item.get("title"), date,
                    url, item.get("content"), year,
                ))
    return output


GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def gdelt_articles(session, params: dict, attempts: int = 4) -> list[dict]:
    """Query GDELT, waiting out its one-request-per-5-seconds rate limit.

    When throttled, GDELT answers with HTTP 200 and a plain-text notice instead
    of JSON, so the body is inspected before parsing and retried with a longer
    pause each round.
    """
    for attempt in range(attempts):
        if attempt:
            time.sleep(6.5 * attempt)
        response = session.get(GDELT_API, params=params, timeout=45)
        response.raise_for_status()
        text = response.text.lstrip()
        if not text:
            return []
        if text[0] in "[{":
            data = response.json()
            return data.get("articles") or [] if isinstance(data, dict) else []
    raise ValueError("GDELT is still rate-limiting this client")


def search_gdelt_source(session, media: str, year: int, pages: int) -> list[dict]:
    """Discover official URLs using one low-frequency domain-limited query."""
    domain, allowed_domains = GDELT_SOURCES[media]
    words = "婚姻 OR 结婚 OR 婚恋 OR 彩礼 OR 生育 OR 出生率 OR 生育率 OR 育儿"
    params = {
        "query": f"domainis:{domain} ({words})", "mode": "ArtList",
        "format": "json", "sort": "DateDesc",
        "maxrecords": min(max(pages * 20, 20), 250),
        "startdatetime": f"{year}0101000000", "enddatetime": f"{year}1231235959",
    }
    output = []
    for item in gdelt_articles(session, params):
        url = item.get("url", "")
        host = urlparse(url).netloc.lower()
        if not any(host == d or host.endswith("." + d) for d in allowed_domains):
            continue
        date = parse_date(url) or parse_date(item.get("seendate"))
        if date and date.year == year:
            output.append(make_candidate(
                media, "combined_topic_query", item.get("title"), date, url, "", year,
            ))
    return output


def extract_article(session, item: dict) -> dict:
    """Download an article and extract title, date, and main text."""
    # Shorter connect budget: unresponsive subdomains (a few People.cn mirrors)
    # would otherwise stall the run for minutes each and are retried next run.
    response = session.get(item["url"], timeout=(12, 30))
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    # Several sites (People.cn among them) ship an empty <h1> container with the
    # real headline in <title>, so every source is tried and the first one that
    # is non-empty wins instead of stopping at the first tag that exists.
    title_meta = soup.find("meta", attrs={"property": "og:title"})
    title_candidates = [title_meta.get("content") if title_meta else ""]
    title_candidates.extend(
        tag.get_text(" ", strip=True)
        for tag in (soup.find("h1"), soup.find("title"))
        if tag
    )
    title_candidates.append(item.get("search_title"))
    title = next((text for text in map(clean_text, title_candidates) if text), "")
    date_values = [item.get("search_date"), item.get("url")]
    for attrs in (
        {"property": "article:published_time"}, {"name": "pubdate"},
        {"name": "publishdate"}, {"name": "PubDate"},
        {"itemprop": "datePublished"}, {"name": "date"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag:
            date_values.append(tag.get("content"))
    date = next((parsed for value in date_values if (parsed := parse_date(value))), None)

    for unwanted in soup(["script", "style", "nav", "footer", "noscript", "form"]):
        unwanted.decompose()
    if item.get("media") in EPAPER_MEDIA:
        # Digital-edition stories live in #ozoom; the generic ".content" wrapper
        # on those pages also pulls in plate navigation.
        selectors = ["#ozoom", "#articleContent"]
    else:
        selectors = [
            "article", "main", "#detail", "#content", "#p-detail", ".TRS_Editor",
            ".article-content", ".article_content", ".article", ".content",
            ".text_con", ".content_area", ".main-aticle",
        ]
    blocks = []
    for selector in selectors:
        for block in soup.select(selector):
            paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in block.find_all("p")]
            text = "\n".join(p for p in paragraphs if len(p) >= 20)
            if text:
                blocks.append(text)
    if not blocks:
        paragraphs = [clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
        blocks = ["\n".join(p for p in paragraphs if len(p) >= 20)]
    text = max(blocks, key=len, default="")
    if len(text) < 200:
        text = item.get("search_content", "")
    return {"title": title, "date": date, "text": text}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Replace CSV atomically; a failed write leaves the old file untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


class StateStore:
    """Crash-safe incremental state using Python's built-in SQLite."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS search_jobs (
                task_key TEXT PRIMARY KEY, media TEXT, keyword TEXT, study_year INTEGER,
                max_pages INTEGER, status TEXT, found INTEGER, error TEXT, updated TEXT
            );
            CREATE TABLE IF NOT EXISTS candidates (
                url TEXT PRIMARY KEY, media TEXT, search_keyword TEXT, search_title TEXT,
                search_date TEXT, search_content TEXT, study_year INTEGER
            );
            CREATE TABLE IF NOT EXISTS processed (
                url TEXT PRIMARY KEY, media TEXT, status TEXT, reason TEXT, updated TEXT
            );
            CREATE TABLE IF NOT EXISTS articles (
                url TEXT PRIMARY KEY, media TEXT, date TEXT, title TEXT, text TEXT,
                text_length INTEGER, search_keywords TEXT, matched_groups TEXT,
                matched_keywords TEXT
            );
        """)
        self.db.commit()

    def import_existing_csv(self, path: Path) -> int:
        """Import an existing corpus once so old work is never overwritten."""
        if not path.exists():
            return 0
        before = self.article_count()
        try:
            with path.open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    if not row.get("url"):
                        continue
                    self.db.execute(
                        """INSERT OR IGNORE INTO articles
                        (url, media, date, title, text, text_length, search_keywords,
                         matched_groups, matched_keywords) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        tuple(row.get(field, "") for field in (
                            "url", "media", "date", "title", "text", "text_length",
                            "search_keywords", "matched_groups", "matched_keywords",
                        )),
                    )
                    self.db.execute(
                        "INSERT OR REPLACE INTO processed VALUES (?, ?, 'retained', '', ?)",
                        (row["url"], row.get("media", ""), now()),
                    )
            self.db.commit()
        except (OSError, csv.Error) as error:
            print(f"WARNING: could not import existing CSV: {error}", flush=True)
        return self.article_count() - before

    def job_completed(self, key: str) -> bool:
        row = self.db.execute(
            "SELECT status FROM search_jobs WHERE task_key=?", (key,)
        ).fetchone()
        return bool(row and row[0] == "completed")

    def save_job(self, key, media, keyword, year, pages, status, found=0, error=""):
        self.db.execute(
            "INSERT OR REPLACE INTO search_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, media, keyword, year, pages, status, found, error, now()),
        )
        self.db.commit()

    def save_candidates(self, rows: list[dict]):
        for row in rows:
            old = self.db.execute(
                "SELECT search_keyword FROM candidates WHERE url=?", (row["url"],)
            ).fetchone()
            if old:
                terms = set((old[0] or "").split("|")) | set(row["search_keyword"].split("|"))
                self.db.execute(
                    "UPDATE candidates SET search_keyword=? WHERE url=?",
                    ("|".join(sorted(term for term in terms if term)), row["url"]),
                )
            else:
                self.db.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                    tuple(row[field] for field in (
                        "url", "media", "search_keyword", "search_title",
                        "search_date", "search_content", "study_year",
                    )),
                )
        self.db.commit()

    def pending_candidates(self, years: list[int]) -> list[dict]:
        marks = ",".join("?" for _ in years)
        rows = self.db.execute(
            f"""SELECT c.media, c.search_keyword, c.search_title, c.search_date,
                       c.url, c.search_content, c.study_year
                FROM candidates c LEFT JOIN processed p ON c.url=p.url
                WHERE c.study_year IN ({marks})
                  AND (p.status IS NULL OR p.status='failed')
                ORDER BY c.study_year, c.media, c.url""",
            years,
        ).fetchall()
        return [dict(zip(CANDIDATE_FIELDS, row)) for row in rows]

    def save_article(self, row: dict):
        self.db.execute(
            """INSERT OR REPLACE INTO articles
            (url, media, date, title, text, text_length, search_keywords,
             matched_groups, matched_keywords) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(row[field] for field in (
                "url", "media", "date", "title", "text", "text_length",
                "search_keywords", "matched_groups", "matched_keywords",
            )),
        )
        self.save_processed(row["url"], row["media"], "retained", "")

    def save_processed(self, url, media, status, reason):
        self.db.execute(
            "INSERT OR REPLACE INTO processed VALUES (?, ?, ?, ?, ?)",
            (url, media, status, reason, now()),
        )
        self.db.commit()

    def article_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    def export(self, output: Path):
        rows = self.db.execute(
            """SELECT media, date, title, text, text_length, url, search_keywords,
                      matched_groups, matched_keywords FROM articles
               ORDER BY date, media, url"""
        ).fetchall()
        write_csv_atomic(output, OUTPUT_FIELDS, [dict(zip(OUTPUT_FIELDS, row)) for row in rows])
        fields = ["media", "url", "reason", "updated"]
        failures = self.db.execute(
            "SELECT media, url, reason, updated FROM processed WHERE status='failed' ORDER BY updated"
        ).fetchall()
        write_csv_atomic(
            output.with_name(output.stem + "_failures.csv"), fields,
            [dict(zip(fields, row)) for row in failures],
        )


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # Ignore stale proxy variables inherited by VS Code.
    session.headers.update(HEADERS)
    retry = Retry(
        total=3, connect=1, read=1, status=3, backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def parse_years(args) -> list[int]:
    if args.start_year is None and args.end_year is None:
        # Running from VS Code without arguments collects the 2021 study year only.
        return [args.year] if args.year is not None else [2021]
    start = args.start_year if args.start_year is not None else (args.year or args.end_year)
    end = args.end_year if args.end_year is not None else start
    if start > end:
        raise SystemExit("--start-year cannot be later than --end-year")
    return list(range(start, end + 1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="Single study year; default without year flags: 2021")
    parser.add_argument("--start-year", type=int, help="First year of a multi-year run")
    parser.add_argument("--end-year", type=int, help="Last year of a multi-year run")
    parser.add_argument(
        "--max-pages", type=int, default=20,
        help=(
            "Maximum search pages per keyword for keyword-searchable sources "
            "(default: 20). Ignored by date-addressed archives. CCTV is swept "
            "in descending date order and uses at least 60 pages, so very "
            "high-frequency keywords may still not reach the study year."
        ),
    )
    parser.add_argument("--max-keywords", type=int, default=0, help="0 means all keywords")
    parser.add_argument("--max-articles", type=int, default=0, help="0 means all pending articles")
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--output", type=Path, help="CSV path; for multiple years use {year} or an automatic year suffix")
    parser.add_argument("--sources", help="Comma-separated names; omit for every source")
    args = parser.parse_args(argv)
    years = parse_years(args)

    # Finish one year before starting the next. Each year has its own CSV and
    # SQLite database; no combined file can overwrite the existing 2025 corpus.
    if len(years) > 1:
        print(f"Annual collection: {years}. Each year is saved separately.", flush=True)
        for year in years:
            print(f"\n===== Starting study year {year} =====", flush=True)
            annual_args = [
                "--year", str(year), "--max-pages", str(args.max_pages),
                "--max-keywords", str(args.max_keywords),
                "--max-articles", str(args.max_articles), "--delay", str(args.delay),
            ]
            if args.sources:
                annual_args.extend(["--sources", args.sources])
            if args.output:
                output_text = str(args.output)
                annual_output = (
                    Path(output_text.replace("{year}", str(year)))
                    if "{year}" in output_text else
                    args.output.with_name(f"{args.output.stem}_{year}{args.output.suffix or '.csv'}")
                )
                annual_args.extend(["--output", str(annual_output)])
            main(annual_args)
        return

    script_dir = Path(__file__).resolve().parent
    label = str(years[0])
    if args.output is None:
        args.output = script_dir / f"official_media_articles_{label}.csv"
    elif not args.output.is_absolute():
        args.output = script_dir / args.output
    args.output = Path(str(args.output).replace("{year}", label))
    state_path = args.output.with_name(args.output.stem + "_state.sqlite3")

    native_searchers = {
        "People.cn": search_people,
        "CCTV": search_cctv,
        "China Daily": search_chinadaily,
        "Workers' Daily / Workercn": search_workercn,
    }
    # Outlets whose 2021 material is only reachable through a date-addressed
    # archive (daily site archive or printed-edition index) rather than a
    # keyword search.
    listing_searchers = {
        "China News Service": search_chinanews_day,
        "China Youth Daily / youth.cn": search_youth_day,
        "Guangming Daily / GMW.cn": search_gmw_day,
    }
    all_sources = list(native_searchers) + list(listing_searchers) + list(GDELT_SOURCES)
    # Project default; extra outlets remain reachable through --sources.
    default_sources = [
        "People.cn", "CCTV", "China News Service", "China Daily",
        "China Youth Daily / youth.cn", "Xinhua", "Guangming Daily / GMW.cn",
    ]
    selected = [name for name in default_sources if name in all_sources]
    if args.sources:
        selected = [value.strip() for value in args.sources.split(",") if value.strip()]
        unknown = sorted(set(selected) - set(all_sources))
        if unknown:
            raise SystemExit(
                f"Unknown source(s): {', '.join(unknown)}\nAvailable: {', '.join(all_sources)}"
            )

    store = StateStore(state_path)
    imported = store.import_existing_csv(args.output)
    store.export(args.output)
    print(f"Output: {args.output}", flush=True)
    print(f"Incremental state: {state_path}", flush=True)
    print(f"Existing articles protected: {store.article_count()} (newly imported: {imported})", flush=True)
    print(f"Years: {years[0]}-{years[-1]} | Sources: {len(selected)}", flush=True)

    session = make_session()
    keywords = QUERY_KEYWORDS[:args.max_keywords] if args.max_keywords else QUERY_KEYWORDS
    tasks = []
    for year in years:
        for media in selected:
            if media in native_searchers:
                tasks.extend((media, keyword, year) for keyword in keywords)
            else:
                # Date-addressed archives and GDELT need a single yearly task:
                # listing outlets sweep every day, GDELT runs one domain query.
                tasks.append((media, "combined_topic_query", year))

    print(f"Search tasks in this configuration: {len(tasks)}", flush=True)
    for number, (media, keyword, year) in enumerate(tasks, start=1):
        key = f"{media}|{keyword}|{year}|pages={args.max_pages}"
        if store.job_completed(key):
            print(f"[Search {number}/{len(tasks)}] SKIP completed | {media} | {keyword} | {year}", flush=True)
            continue
        print(f"[Search {number}/{len(tasks)}] {media} | {keyword} | {year}", flush=True)
        store.save_job(key, media, keyword, year, args.max_pages, "running")
        day_errors = 0
        try:
            if media in native_searchers:
                rows = native_searchers[media](session, keyword, year, args.max_pages)
            elif media in listing_searchers:
                # Walk the date-addressed archive day by day. A single day's
                # transient timeout must not discard the whole year, so each day
                # is guarded and the year is re-scanned later to fill any gap.
                rows = []
                for day in year_days(year):
                    try:
                        rows.extend(listing_searchers[media](session, day, keywords))
                    except requests.RequestException as error:
                        day_errors += 1
                        print(f"  Skipped {day:%Y-%m-%d} after a network error: {error}", flush=True)
                    time.sleep(args.delay)
            else:
                rows = search_gdelt_source(session, media, year, args.max_pages)
            store.save_candidates(rows)
            if day_errors:
                raise requests.RequestException(
                    f"{day_errors} archive day(s) failed; the year will be re-scanned"
                )
            store.save_job(key, media, keyword, year, args.max_pages, "completed", len(rows))
            print(f"  Saved {len(rows)} candidate(s).", flush=True)
        except Exception as error:
            store.save_job(key, media, keyword, year, args.max_pages, "failed", 0, str(error))
            print(f"  FAILED (will retry next run): {error}", flush=True)
        time.sleep(args.delay)

    pending = store.pending_candidates(years)
    if args.max_articles:
        pending = pending[:args.max_articles]
    print(f"Pending candidate articles: {len(pending)}", flush=True)
    retained_this_run = 0
    for number, item in enumerate(pending, start=1):
        print(f"[Download {number}/{len(pending)}] {item['media']} | {item['url']}", flush=True)
        try:
            article = extract_article(session, item)
            if not article["date"] or article["date"].year != item["study_year"]:
                store.save_processed(item["url"], item["media"], "excluded", "outside_study_year")
                print("  Excluded: date outside the study year.", flush=True)
                continue
            if len(article["text"]) < 200:
                store.save_processed(item["url"], item["media"], "excluded", "short_text")
                print("  Excluded: article text is too short.", flush=True)
                continue
            if item["media"] == "China Daily" and not is_mainly_chinese(article["text"]):
                store.save_processed(item["url"], item["media"], "excluded", "not_mainly_chinese")
                print("  Excluded: China Daily article is not mainly Chinese.", flush=True)
                continue
            groups, words = matches(f"{article['title']} {article['text']}")
            if not words:
                store.save_processed(item["url"], item["media"], "excluded", "no_topic_keyword")
                print("  Excluded: no study keyword in the full article.", flush=True)
                continue
            row = {
                "media": item["media"], "date": article["date"].strftime("%Y-%m-%d"),
                "title": article["title"], "text": article["text"],
                "text_length": len(article["text"]), "url": item["url"],
                "search_keywords": item["search_keyword"],
                "matched_groups": "|".join(groups), "matched_keywords": "|".join(words),
            }
            store.save_article(row)
            retained_this_run += 1
            print(f"  Retained and checkpointed ({retained_this_run} new).", flush=True)
        except Exception as error:
            store.save_processed(item["url"], item["media"], "failed", str(error))
            print(f"  FAILED (will retry next run): {error}", flush=True)
        if retained_this_run and retained_this_run % 10 == 0:
            store.export(args.output)
            print("  CSV checkpoint refreshed.", flush=True)
        time.sleep(args.delay)

    store.export(args.output)
    print(f"Finished. Corpus now contains {store.article_count()} articles.", flush=True)
    print(f"CSV: {args.output}", flush=True)
    print("Safe to stop and rerun: completed work will be skipped.", flush=True)


if __name__ == "__main__":
    main()
