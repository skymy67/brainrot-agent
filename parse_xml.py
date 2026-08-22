#!/usr/bin/env python3
"""Convert a MediaWiki Special:Export XML dump into wiki_data.json."""

import json
import re
import xml.etree.ElementTree as ET

import mwparserfromhell

INPUT_FILES = ["Italian+Brainrot+Wiki-20260821163544.xml", "Slop_God.xml"]
OUTPUT_FILE = "wiki_data.json"
BASE_URL = "https://italianbrainrot.wikioasis.org/wiki/"
MAIN_NAMESPACE = "0"
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

MAGIC_WORDS_RE = re.compile(r"__[A-Z]+__")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def to_plain_text(wikitext):
    """Render wikitext down to plain prose, matching the MediaWiki API's explaintext output."""
    text = mwparserfromhell.parse(wikitext).strip_code()
    text = MAGIC_WORDS_RE.sub("", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def parse_pages(path):
    pages = []
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != f"{MW_NS}page":
            continue

        ns = elem.findtext(f"{MW_NS}ns")
        if ns != MAIN_NAMESPACE or elem.find(f"{MW_NS}redirect") is not None:
            elem.clear()
            continue

        title = elem.findtext(f"{MW_NS}title")
        revision = elem.find(f"{MW_NS}revision")
        wikitext = revision.findtext(f"{MW_NS}text") or "" if revision is not None else ""

        pages.append(
            {
                "title": title,
                "content": to_plain_text(wikitext),
                "url": BASE_URL + title.replace(" ", "_"),
            }
        )
        elem.clear()

    return pages


def merge_pages(*page_lists):
    """Merge page lists by title, later lists override earlier ones for the same title."""
    by_title = {}
    for pages in page_lists:
        for page in pages:
            by_title[page["title"]] = page
    return list(by_title.values())


def main():
    pages = merge_pages(*(parse_pages(path) for path in INPUT_FILES))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pages)} pages to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
