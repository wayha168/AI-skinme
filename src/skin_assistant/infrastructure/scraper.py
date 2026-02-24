# Optional HTML scraping with BeautifulSoup (bs4).
# Use for HTML product pages; for JSON APIs use skinme_client instead.

from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore


def fetch_html(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "SkinAssistant/1.0"})
    resp.raise_for_status()
    return resp.text


def parse_html(html: str, parser: str = "html.parser"):
    if BeautifulSoup is None:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")
    return BeautifulSoup(html, parser)


def scrape_table_to_rows(
    url: str,
    table_selector: str = "table",
    row_selector: str = "tr",
    cell_selector: str = "td",
    header_row: bool = True,
    base_url: Optional[str] = None,
) -> list[dict]:
    html = fetch_html(url)
    soup = parse_html(html)
    table = soup.select_one(table_selector)
    if not table:
        return []
    rows = table.select(row_selector)
    if not rows:
        return []
    out = []
    headers = None
    for i, row in enumerate(rows):
        cells = row.select(cell_selector)
        if not cells:
            continue
        texts = [c.get_text(strip=True) for c in cells]
        if header_row and i == 0:
            headers = texts
            continue
        if headers and len(texts) <= len(headers):
            out.append(dict(zip(headers, texts)))
    return out


def save_scraped_to_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
