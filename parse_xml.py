#!/usr/bin/env python3
"""Convert a MediaWiki Special:Export XML dump into wiki_data.json."""

import json
import re
import xml.etree.ElementTree as ET

import mwparserfromhell

INPUT_FILES = ["Italian+Brainrot+Wiki-20260821163544.xml", "Slop_God.xml"]
OUTPUT_FILE = "wiki_data.json"
BASE_URL = "https://italianbrainrot.wikioasis.org/wiki/"
IMAGE_BASE_URL = "https://italianbrainrot.wikioasis.org/wiki/Special:FilePath/"
MAIN_NAMESPACE = "0"
MW_NS = "{http://www.mediawiki.org/xml/export-0.11/}"

MAGIC_WORDS_RE = re.compile(r"__[A-Z]+__")
BLANK_LINES_RE = re.compile(r"\n{3,}")

# Every alias this wiki's editors use for the character infobox template — found by scanning
# every template with an "image" parameter across the full dump: "Brainrot" is the primary name,
# "B"/"Brainrot2/3/4" are apparent copy-pasted variants, and "Infobox" is a small minority using
# the generic MediaWiki template name directly instead.
INFOBOX_TEMPLATE_NAMES = {"brainrot", "b", "brainrot2", "brainrot3", "brainrot4", "infobox"}

GALLERY_OPEN_RE = re.compile(r"(?i)^<gallery")
# A gallery line's filename/caption separator is sometimes a literal "|", sometimes the
# "{{!}}" pipe-escape template editors use to avoid it being parsed as a new template param.
CAPTION_SEP_RE = re.compile(r"\{\{!\}\}|\|")
# Means "use this page's own title as the filename" — a handful of pages use this as a
# template instead of hardcoding their filename directly.
PAGENAME_RE = re.compile(r"(?i)\{\{\s*(?:FULL)?PAGENAME\s*\}\}")
# A handful of pages have a "<gallery>" typo'd backwards (as "</gallery>") or doubled
# ("<<gallery>>") as their very first line — rare enough it's not worth pattern-matching every
# variant; this catches all of them generically by rejecting whatever gets extracted afterward
# if it doesn't actually look like an image filename, rather than storing an image_url that
# would 404.
LOOKS_LIKE_FILENAME_RE = re.compile(r"(?i)^[^<>{}|\n]+\.(png|jpe?g|gif|webp|svg|bmp|mp4|gifv)$")

# The wiki's own first-party content-warning template — editors attach this (with a "details"/
# "severity" note) to pages whose song lyrics contain extreme profanity, blasphemy, or sexual
# content (verified by inspecting real flagged pages, e.g. Tralalero Tralala's lyrics literally
# translate to "damn God and damn Allah" plus much more explicit material). ~650 of the wiki's
# ~4700 pages carry this. "nooffensiveban" is an unrelated anti-vandalism template found on a
# User: talk page during the audit below and deliberately excluded — parse_pages() only processes
# the main (character) namespace anyway, so it would never match a real page, but it's excluded
# here too for clarity. Same "trust the wiki's own first-party signal" approach already used for
# INFOBOX_TEMPLATE_NAMES above and akinator_mode.py's Category:Baby handling, rather than this
# pipeline trying to judge "is this offensive" itself.
OFFENSIVE_TEMPLATE_RE = re.compile(r"\{\{\s*(?:offensive|offensive2|offensive content)\s*[|}]", re.IGNORECASE)

# Matches a whole "== Lyrics ==" section — the heading line through everything up to (but not
# including) the next level-2 heading or end of page. Operates on the RAW wikitext string, before
# mwparserfromhell ever parses it — deliberately, not a tree-based mwparserfromhell.get_sections()
# pass: on real pages here, an unclosed/malformed {{Collapse|...}} template wrapping the giant
# multi-language lyrics dump confuses mwparserfromhell's tree parser (both get_sections() and a
# manual top-level node scan were tested and both silently swallowed the following History/
# Etymology sections into the malformed template's fallout — verified against Tralalero Tralala's
# real page). A level-2 heading in this wiki's convention is always written as a line that starts
# "==" followed by something other than a third "=" (which would make it a level-3+ subheading
# instead) — a plain string match on that exact convention isn't tricked by the same malformed
# markup that fools structural parsing.
LYRICS_SECTION_RE = re.compile(r"(?ms)^==\s*Lyrics\s*==\s*\n.*?(?=^==[^=]|\Z)")


def to_plain_text(code):
    """Render already-parsed wikitext down to plain prose, matching the MediaWiki API's
    explaintext output."""
    text = code.strip_code()
    text = MAGIC_WORDS_RE.sub("", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _normalize_filename_candidate(text):
    """Cleans up one raw filename candidate down to a bare filename: drops a wikilink's opening
    "[[", any trailing caption after a "|" or "{{!}}" separator (plus a wikilink's closing "]]",
    which only shows up once the caption itself has been split off), and a File:/Image:
    namespace prefix — that prefix belongs to the wikilink syntax, not the actual file on disk,
    so leaving it in would build a URL for a file that doesn't exist."""
    text = text.strip().lstrip("[")  # tolerates an unbalanced single "[" typo, not just "[["
    text = CAPTION_SEP_RE.split(text)[0]
    text = text.rstrip("]").strip()
    if text.lower().startswith(("file:", "image:")):
        text = text.split(":", 1)[1].strip()
    return text


def _first_gallery_filename(text):
    """Pulls the first real filename out of gallery-block content: one filename per line,
    optionally File:/Image:-prefixed, optionally followed by a caption after a separator."""
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("<!--") or GALLERY_OPEN_RE.match(line) or line == "</gallery>":
            continue
        filename = _normalize_filename_candidate(line)
        if filename:
            return filename
    return None


def extract_image_filename(code, title):
    """Finds the character's portrait filename from parsed wikitext, in three tiers (99.7%
    combined coverage measured against the full dump): (1) the "image" parameter of the page's
    infobox template — the normal case; (2) if that's empty, the first filename inside a
    <gallery> block, a pattern some pages use instead of the image parameter; (3) if there's no
    recognized infobox at all, the first bare [[File:...]]/[[Image:...]] wikilink anywhere on the
    page. Returns None (no image_url is stored) rather than guessing when nothing is found —
    better to omit an image than link a page to the wrong one. Validates whatever the tiers
    above come up with against LOOKS_LIKE_FILENAME_RE before returning it, so a handful of truly
    bizarre one-off wikitext typos (a backwards "</gallery>", a doubled "<<gallery>>") can't slip
    a garbage non-filename string through as a broken image_url."""
    filename = _extract_image_filename_unvalidated(code, title)
    return filename if filename and LOOKS_LIKE_FILENAME_RE.match(filename) else None


def _extract_image_filename_unvalidated(code, title):
    for template in code.filter_templates():
        name = str(template.name).strip().lower()
        if name not in INFOBOX_TEMPLATE_NAMES:
            continue
        if template.has("image"):
            image_value = template.get("image").value  # a Wikicode fragment, not plain text

            # A handful of pages leave the "image" param blank and put a <gallery> block
            # directly inside it instead of after the template — has to be found as a nested
            # tag in the parsed fragment, not string-matched, or its raw "<gallery>" markup
            # gets mistaken for the filename itself.
            nested_galleries = list(image_value.filter_tags(matches=lambda t: t.tag == "gallery"))
            if nested_galleries:
                found = _first_gallery_filename(str(nested_galleries[0].contents))
                if found:
                    return found
            else:
                value = str(image_value).strip()
                if GALLERY_OPEN_RE.match(value):
                    # A <gallery> tag missing its closing </gallery> — mwparserfromhell can't
                    # form a proper Tag node for an unclosed tag, so this falls into the plain
                    # else branch above with the whole malformed blob (including whatever
                    # later infobox fields got swallowed by the missing close) as one string.
                    # Pull the filename out of the raw text by hand instead.
                    found = _first_gallery_filename("\n".join(value.splitlines()[1:]))
                    if found:
                        return found
                else:
                    # Some pages put a stray caption/size param after another separator on
                    # the same value, or even a second filename on its own line below a
                    # malformed first one — the text before the first separator on the first
                    # line is the actual filename in every real case.
                    first_line = value.splitlines()[0].strip() if value else ""
                    first_line = PAGENAME_RE.sub(title.replace(" ", "_"), first_line)
                    filename = _normalize_filename_candidate(first_line)
                    if filename:
                        return filename
        break  # only the page's own (first) infobox counts, not an infobox on a linked page

    for tag in code.filter_tags(matches=lambda t: t.tag == "gallery"):
        found = _first_gallery_filename(str(tag.contents))
        if found:
            return found

    for link in code.filter_wikilinks():
        wikilink_title = str(link.title).strip()
        if wikilink_title.lower().startswith(("file:", "image:")):
            return _normalize_filename_candidate(wikilink_title)

    return None


def _is_flagged_offensive(wikitext):
    return bool(OFFENSIVE_TEMPLATE_RE.search(wikitext))


def _strip_lyrics_section(wikitext):
    """Removes a flagged page's whole "Lyrics" section — heading plus every nested subsection
    (each language's translation) — from the raw wikitext, before it's ever parsed, flattened to
    plain text, or stored. That's specifically where the explicit/blasphemous content actually
    lives on every flagged page checked; everything else (description, history, reception,
    trivia, battle info) is left untouched, so one section doesn't cost a character its entire
    usable content — the resulting prompt no longer trips Gemini's own safety filter on the
    character's own name."""
    return LYRICS_SECTION_RE.sub("", wikitext)


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
        if _is_flagged_offensive(wikitext):
            wikitext = _strip_lyrics_section(wikitext)
        code = mwparserfromhell.parse(wikitext)

        image_filename = extract_image_filename(code, title)
        page = {
            "title": title,
            "content": to_plain_text(code),
            "url": BASE_URL + title.replace(" ", "_"),
        }
        if image_filename:
            page["image_url"] = IMAGE_BASE_URL + image_filename.replace(" ", "_")
        pages.append(page)
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
