#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote_from_bytes, unquote, urljoin, urlparse

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
    ],
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

# GDELT only discovers URLs for sites without a stable non-browser search API.
# Article text is always downloaded from the official domain itself.
GDELT_SOURCES = {}  # Not used: this study searches only the three native sites.


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


def checkpoint_search_page(session, page: int, rows: list[dict], exhausted=False):
    """Persist one fully parsed page before requesting the next one."""
    callback = getattr(session, "page_checkpoint", None)
    if callback:
        callback(page, rows, exhausted)


def search_people(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search the People's Daily website."""
    endpoints = [
        "https://search.people.cn/search-platform/front/search",
        "https://search.people.cn/api-search/front/search",
        "http://search.people.cn/search-platform/front/search",
    ]
    output, working_endpoint = [], None
    for page in range(getattr(session, "resume_page", 1), pages + 1):
        before = len(output)
        payload = {
            "key": keyword, "page": page, "limit": 10, "sortType": 2,
            "type": 0, "hasTitle": True, "hasContent": True,
            "isFuzzy": True, "startTime": 0, "endTime": 0,
        }
        data = None
        for endpoint in ([working_endpoint] if working_endpoint else endpoints):
            try:
                response = session.post(endpoint, json=payload, timeout=25)
                response.raise_for_status()
                possible = response.json()
                if possible.get("data") is not None:
                    data, working_endpoint = possible, endpoint
                    break
            except (requests.RequestException, ValueError):
                continue
        if data is None:
            raise requests.RequestException("All People.cn search endpoints failed")
        records = (data.get("data") or {}).get("records") or []
        if not records:
            checkpoint_search_page(session, page, [], exhausted=True)
            break
        for item in records:
            date = parse_date(item.get("displayTime"))
            if date and date.year == year and item.get("url"):
                output.append(make_candidate(
                    "People.cn", keyword, item.get("title"), date,
                    item["url"], item.get("content"), year,
                ))
        checkpoint_search_page(session, page, output[before:])
    return output


def search_cctv(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search CCTV News."""
    output = []
    for page in range(getattr(session, "resume_page", 1), pages + 1):
        before = len(output)
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
            checkpoint_search_page(session, page, [], exhausted=True)
            break
        for link in links:
            href = link.get("href", "")
            target = parse_qs(urlparse(href).query).get("targetpage", [None])[0]
            url = unquote(target) if target else href
            date = nearby_date(link) or parse_date(url)
            if "news.cctv.com" in urlparse(url).netloc.lower() and date and date.year == year:
                output.append(make_candidate("CCTV", keyword, link.get_text(), date, url, "", year))
        checkpoint_search_page(session, page, output[before:])
    return output


def read_search_variable(html: str, name: str):
    """Decode embedded JSON without executing the page's JavaScript."""
    match = re.search(r"\bvar\s+" + re.escape(name) + r"\s*=\s*", html)
    if not match:
        raise ValueError(f"China News Service page is missing {name}; not a valid empty result")
    try:
        value, _ = json.JSONDecoder().raw_decode(html[match.end():])
        return value
    except ValueError as error:
        raise ValueError(f"Cannot decode China News Service variable {name}") from error


def search_chinanews(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Read embedded docArr results from the site's current search form."""
    output = []
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"
    for page in range(getattr(session, "resume_page", 1), pages + 1):
        response = session.post(
            "https://sou.chinanews.com.cn/search/news",
            data={
                "q": keyword, "searchField": "all", "sortType": "time",
                "dateType": "", "startDate": start_date, "endDate": end_date,
                "channel": "all", "pageNum": str(page),
            },
            timeout=25,
        )
        response.raise_for_status()
        html = response.text
        # Detect ignored filters or pagination rather than checkpointing wrong results.
        if (
            read_search_variable(html, "startDate") != start_date
            or read_search_variable(html, "endDate") != end_date
            or read_search_variable(html, "pageNum") != page
        ):
            raise ValueError("China News Service ignored the requested date range or page")
        records = read_search_variable(html, "docArr")
        total = read_search_variable(html, "totalNum")
        if not isinstance(records, list) or not isinstance(total, int) or total < 0:
            raise ValueError("Invalid China News Service search-result structure")
        if not records:
            if total > 0 and page <= (total + 9) // 10:
                raise ValueError("China News Service reports matches but returned no records")
            checkpoint_search_page(session, page, [], exhausted=True)
            break
        page_rows = []
        for item in records:
            if not isinstance(item, dict):
                raise ValueError("Invalid China News Service article record")
            url = urljoin("https://www.chinanews.com.cn/", item.get("url") or "")
            host = urlparse(url).netloc.lower()
            raw_title = item.get("title")
            if isinstance(raw_title, list):
                raw_title = " ".join(str(value) for value in raw_title if value)
            title = clean_text(raw_title)
            # pubtime may be a later update; the dated URL identifies the
            # original article, followed by the index's creation timestamp.
            date = parse_date(url) or parse_date(item.get("createtime")) or parse_date(item.get("pubtime"))
            if date and date.year != year:
                raise ValueError("China News Service returned articles outside the requested year")
            if (
                any(host == d or host.endswith("." + d) for d in ("chinanews.com.cn", "chinanews.com"))
                and date and date.year == year and title
                and re.search(r"\.s?html(?:\?|$)", url)
            ):
                page_rows.append(make_candidate(
                    # Require article-page text; do not substitute search snippets.
                    "China News Service", keyword, title, date, url, "", year,
                ))
        if not page_rows:
            raise ValueError("China News Service returned records, but none could be parsed")
        output.extend(page_rows)
        exhausted = page * 10 >= total
        checkpoint_search_page(session, page, page_rows, exhausted=exhausted)
        if exhausted:
            break
    return output


def search_chinadaily(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search Chinese-language China Daily through its JSON API."""
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
    for page in range(pages):
        response = session.get(
            "https://newssearch.chinadaily.com.cn/rest/cn/search",
            params={"keywords": keyword, "sort": "dp", "page": page, "curType": "story"},
            headers=ajax_headers,
            timeout=30,
        )
        response.raise_for_status()
        records = response.json().get("content") or []
        if not records:
            break
        for item in records:
            url = item.get("url", "")
            host = urlparse(url).netloc.lower()
            date = parse_date(item.get("publishTime")) or parse_date(url)
            if host.endswith("chinadaily.com.cn") and date and date.year == year:
                output.append(make_candidate(
                    "China Daily", keyword, item.get("title"), date, url,
                    item.get("plainText") or item.get("content"), year,
                ))
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


def unwrap_youth_url(url: str) -> str:
    marker = "/transfer/index/url/"
    if marker in url:
        target = unquote(url.split(marker, 1)[1])
        return target if target.startswith("http") else "https://" + target
    return url


def search_youth(session, keyword: str, year: int, pages: int) -> list[dict]:
    """Search China Youth Daily / youth.cn; exclude user-generated pages."""
    output = []
    encoded = quote_from_bytes(keyword.encode("gb2312", errors="ignore"))
    for page in range(pages):
        url = (
            "http://search.youth.cn/cse/search?"
            f"s=15107678543080134641&entry=1&ie=gb2312&q={encoded}&p={page}"
        )
        response = session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        page_rows = []
        for link in soup.find_all("a", href=True):
            article_url = unwrap_youth_url(urljoin(response.url, link["href"]))
            host = urlparse(article_url).netloc.lower()
            date = parse_date(article_url) or nearby_date(link)
            title = clean_text(link.get_text())
            if (
                host.endswith("youth.cn") and host != "user.youth.cn"
                and date and date.year == year and title
                and re.search(r"\.s?htm(?:l)?(?:\?|$)", article_url)
            ):
                page_rows.append(make_candidate(
                    "China Youth Daily / youth.cn", keyword, title, date,
                    article_url, "", year,
                ))
        if not page_rows:
            break
        output.extend(page_rows)
    return output


def search_gdelt_source(session, media: str, year: int, pages: int) -> list[dict]:
    """Discover official URLs using one low-frequency domain-limited query."""
    domain, allowed_domains = GDELT_SOURCES[media]
    words = " OR ".join(QUERY_KEYWORDS)
    response = session.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={
            "query": f"domainis:{domain} ({words})", "mode": "ArtList",
            "format": "json", "sort": "DateDesc",
            "maxrecords": min(max(pages * 20, 20), 250),
            "startdatetime": f"{year}0101000000", "enddatetime": f"{year}1231235959",
        },
        timeout=45,
    )
    response.raise_for_status()
    output = []
    for item in response.json().get("articles", []):
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


def extract_title(soup, search_title: str = "") -> str:
    """Use the first non-empty title; an empty h1 must not block fallbacks."""
    candidates = []
    for attrs in ({"property": "og:title"}, {"name": "twitter:title"}):
        candidates.extend(tag.get("content") for tag in soup.find_all("meta", attrs=attrs))
    candidates.extend(tag.get_text(" ", strip=True) for tag in soup.find_all("h1"))
    candidates.extend(tag.get_text(" ", strip=True) for tag in soup.find_all("title"))
    candidates.append(search_title)
    for value in candidates:
        title = clean_text(value)
        if title:
            return title
    return ""  # Never fabricate a title when every source is empty.


def extract_article(session, item: dict) -> dict:
    """Download an article and extract title, date, and main text."""
    response = session.get(item["url"], timeout=35)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    title = extract_title(soup, item.get("search_title", ""))
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
            CREATE TABLE IF NOT EXISTS search_progress (
                media TEXT, keyword TEXT, study_year INTEGER,
                next_page INTEGER, exhausted INTEGER, updated TEXT,
                PRIMARY KEY (media, keyword, study_year)
            );
            CREATE TABLE IF NOT EXISTS collector_versions (
                media TEXT PRIMARY KEY, version TEXT
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

    def page_progress(self, media, keyword, year):
        row = self.db.execute(
            "SELECT next_page, exhausted FROM search_progress WHERE media=? AND keyword=? AND study_year=?",
            (media, keyword, year),
        ).fetchone()
        return (row[0], bool(row[1])) if row else (1, False)

    def ensure_search_version(self, media, version):
        """Invalidate only this source's old search checkpoints once per fix."""
        row = self.db.execute(
            "SELECT version FROM collector_versions WHERE media=?", (media,),
        ).fetchone()
        if row and row[0] == version:
            return
        with self.db:
            count = self.db.execute("SELECT COUNT(*) FROM search_jobs WHERE media=?", (media,)).fetchone()[0]
            self.db.execute("DELETE FROM search_jobs WHERE media=?", (media,))
            self.db.execute("DELETE FROM search_progress WHERE media=?", (media,))
            self.db.execute("INSERT OR REPLACE INTO collector_versions VALUES (?, ?)", (media, version))
        print(f"Updated {media} search parser: reset {count} old search task(s); saved articles and other sources kept.", flush=True)

    def save_page(self, media, keyword, year, page, rows, exhausted):
        self.save_candidates(rows)
        self.db.execute(
            "INSERT OR REPLACE INTO search_progress VALUES (?, ?, ?, ?, ?, ?)",
            (media, keyword, year, page + 1, int(exhausted), now()),
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
        total=3, connect=3, read=2, status=3, backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def parse_years(args) -> list[int]:
    if args.start_year is None and args.end_year is None:
        # Running from VS Code without arguments collects 2023-2025.
        return [args.year] if args.year is not None else list(range(2023, 2026))
    start = args.start_year if args.start_year is not None else (args.year or args.end_year)
    end = args.end_year if args.end_year is not None else start
    if start > end:
        raise SystemExit("--start-year cannot be later than --end-year")
    return list(range(start, end + 1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="Single study year; default without year flags: 2023-2025")
    parser.add_argument("--start-year", type=int, help="First year of a multi-year run")
    parser.add_argument("--end-year", type=int, help="Last year of a multi-year run")
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Maximum search pages per keyword/source (default: 50; stops earlier when results run out)",
    )
    parser.add_argument("--max-keywords", type=int, default=0, help="0 means all keywords")
    parser.add_argument("--max-articles", type=int, default=0, help="0 means all pending articles")
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--output", type=Path, help="CSV path; for multiple years use {year} or an automatic year suffix")
    parser.add_argument("--sources", help="Comma-separated names; omit for every source")
    args = parser.parse_args(argv)
    years = parse_years(args)

    # Finish one year before starting the next. Each year has its own CSV and
    # SQLite database in this script's folder, separate from the other versions.
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
    # Do not resume old fertility searches or import an old mixed-topic CSV.
    state_path = args.output.with_name(args.output.stem + "_marriage_three_sources_state.sqlite3")

    native_searchers = {
        "People.cn": search_people,
        "CCTV": search_cctv,
        "China News Service": search_chinanews,
    }
    all_sources = list(native_searchers)
    selected = all_sources
    if args.sources:
        selected = [value.strip() for value in args.sources.split(",") if value.strip()]
        unknown = sorted(set(selected) - set(all_sources))
        if unknown:
            raise SystemExit(
                f"Unknown source(s): {', '.join(unknown)}\nAvailable: {', '.join(all_sources)}"
            )

    store = StateStore(state_path)
    if "China News Service" in selected:
        store.ensure_search_version("China News Service", "docarr_date_page_v1")
    store.export(args.output)
    print(f"Output: {args.output}", flush=True)
    print(f"Incremental state: {state_path}", flush=True)
    print(f"Marriage-only articles already checkpointed: {store.article_count()}", flush=True)
    print(f"Years: {years[0]}-{years[-1]} | Sources: {len(selected)}", flush=True)

    session = make_session()
    try:
        collect_annual(store, args, years, selected, native_searchers, session)
    finally:
        # Ctrl+C still exports every article already committed to SQLite.
        try:
            store.export(args.output)
            print(f"CSV saved: {args.output}", flush=True)
        finally:
            try:
                session.close()
            finally:
                store.db.close()


def collect_annual(store, args, years, selected, native_searchers, session):
    keywords = QUERY_KEYWORDS[:args.max_keywords] if args.max_keywords else QUERY_KEYWORDS
    tasks = []
    for year in years:
        for media in selected:
            tasks.extend((media, keyword, year) for keyword in keywords)

    remaining = []
    for media, keyword, year in tasks:
        key = f"{media}|{keyword}|{year}|pages={args.max_pages}"
        next_page, exhausted = store.page_progress(media, keyword, year)
        if store.job_completed(key) or exhausted or next_page > args.max_pages:
            continue
        remaining.append((key, media, keyword, year, next_page))
    print(f"Search checkpoint: {len(tasks)-len(remaining)} completed, {len(remaining)} remaining.", flush=True)
    for number, (key, media, keyword, year, next_page) in enumerate(remaining, start=1):
        print(f"[Search {number}/{len(remaining)} remaining] {media} | {keyword} | {year} | resume page {next_page}", flush=True)
        store.save_job(key, media, keyword, year, args.max_pages, "running")
        session.resume_page = next_page

        def save_page(page, rows, exhausted):
            store.save_page(media, keyword, year, page, rows, exhausted)
            print(f"  [Page {page}/{args.max_pages}] {len(rows)} candidate(s) saved | next page {page+1}", flush=True)
            time.sleep(args.delay)

        session.page_checkpoint = save_page
        try:
            rows = native_searchers[media](session, keyword, year, args.max_pages)
            store.save_candidates(rows)
            store.save_job(key, media, keyword, year, args.max_pages, "completed", len(rows))
            print(f"  Saved {len(rows)} candidate(s).", flush=True)
        except Exception as error:
            store.save_job(key, media, keyword, year, args.max_pages, "failed", 0, str(error))
            print(f"  FAILED (will retry next run): {error}", flush=True)
        time.sleep(args.delay)

    pending = [item for item in store.pending_candidates(years) if item["media"] in selected]
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
    try:
        main()
    except KeyboardInterrupt:
        print("Paused safely. Rerun to continue unfinished searches and downloads.", flush=True)
