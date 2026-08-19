#!/usr/bin/env python3
"""Convert a MediaWiki Special:Export XML dump into wiki_data.json."""

import json
import xml.etree.ElementTree as ET

INPUT_FILE = "Italian+Brainrot+Wiki-20260819145439.xml"
OUTPUT_FILE = "wiki_data.json"
BASE_URL = "https://italianbrainrot.wikioasis.org/wiki/"
MAIN_NAMESPACE = "0"
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


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
        content = revision.findtext(f"{MW_NS}text") or "" if revision is not None else ""

        pages.append(
            {
                "title": title,
                "content": content,
                "url": BASE_URL + title.replace(" ", "_"),
            }
        )
        elem.clear()

    return pages


def main():
    pages = parse_pages(INPUT_FILE)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pages)} pages to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
