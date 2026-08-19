#!/usr/bin/env python3
"""Scrape all main-namespace pages from the Italian Brainrot wiki via the MediaWiki API."""

import json
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests

API_URL = "https://italianbrainrot.wikioasis.org/api.php"
BASE_URL = "https://italianbrainrot.wikioasis.org/wiki/"
REQUEST_DELAY = 0.5
MAX_RETRIES = 3
OUTPUT_FILE = "wiki_data.json"
SYNC_FILE = "last_sync.json"

session = requests.Session()
session.headers.update({"User-Agent": "brainrot-agent-scraper/1.0"})


def api_get(params, max_retries=MAX_RETRIES):
    """Issue a GET request against the MediaWiki API with retries."""
    params = {**params, "format": "json"}
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"Request failed (attempt {attempt}/{max_retries}): {exc}")
            if attempt == max_retries:
                print(f"Giving up on request: {urlencode(params)}")
                return None
            time.sleep(REQUEST_DELAY * attempt)
    return None


def fetch_all_page_titles():
    """Fetch all page titles in the main namespace, paginating via apcontinue."""
    titles = []
    apcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "allpages",
            "apnamespace": 0,
            "aplimit": "max",
            "apfilterredir": "nonredirects",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue

        data = api_get(params)
        if data is None:
            print("Failed to fetch a page list batch; stopping pagination.")
            break

        pages = data.get("query", {}).get("allpages", [])
        for page in pages:
            titles.append(page["title"])

        print(f"Fetched {len(pages)} titles (total so far: {len(titles)})")

        cont = data.get("continue")
        if not cont or "apcontinue" not in cont:
            break
        apcontinue = cont["apcontinue"]

        time.sleep(REQUEST_DELAY)

    return titles


def fetch_page_content(title):
    """Fetch plain-text content, pageid, and last-edited timestamp for a page."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }

    data = api_get(params)
    if data is None:
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "missing" in page:
        print(f"Page missing, skipping: {title}")
        return None

    return {
        "pageid": page.get("pageid"),
        "title": page.get("title", title),
        "content": page.get("extract", ""),
        "url": BASE_URL + page.get("title", title).replace(" ", "_"),
        "last_updated": page.get("touched"),
    }


def main():
    print("Fetching all page titles...")
    titles = fetch_all_page_titles()
    print(f"Found {len(titles)} pages total.")

    results = []
    for i, title in enumerate(titles, start=1):
        print(f"[{i}/{len(titles)}] Fetching content for: {title}")
        page_data = fetch_page_content(title)
        if page_data is not None:
            results.append(page_data)
        time.sleep(REQUEST_DELAY)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} pages to {OUTPUT_FILE}")

    sync_info = {"last_sync": datetime.now(timezone.utc).isoformat()}
    with open(SYNC_FILE, "w", encoding="utf-8") as f:
        json.dump(sync_info, f, ensure_ascii=False, indent=2)
    print(f"Saved sync timestamp to {SYNC_FILE}")


if __name__ == "__main__":
    main()
