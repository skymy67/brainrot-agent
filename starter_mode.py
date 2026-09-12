#!/usr/bin/env python3
"""Starter Mode: suggests real wiki characters suitable as game "starters" (à la Pokémon
starters) — characters with a genuinely documented evolution chain, variant relationship, or
family/tier progression, well-documented enough to make a solid RPG Mode Dex Entry from later.

Candidate narrowing is entirely code-only and free (no Gemini call, no RAG/embedding call
either): it reuses akinator_mode.py's own Category: tag mining — already done once at startup —
specifically its family/baby-relationship supercategories (Category:Baby/Bambino, and the
wiki's own named family groups like "Sahur Family", "67 Family"). These are first-party wiki
signals of a real documented relationship, the same "trust the wiki's own signal" reasoning
evolution_mode.py and akinator_mode.py already use elsewhere — never an invented or guessed
relationship. A page's content length is used as a cheap, free proxy for "well-documented
enough," and each family tag is capped so one dominant family (Sahur Family alone covers roughly
half of the wiki's qualifying pool) can't crowd out variety.

Exactly ONE Gemini call is made per request — picking, ranking, and describing the final 5-10
shortlist from that pre-narrowed, already-real candidate pool — matching the low-call-count
design every other mode in this app now follows.
"""

import json

from google.genai import types
from pydantic import BaseModel

import akinator_mode
from content_policy import with_content_policy
from gemini_retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 640
MAX_OUTPUT_TOKENS = 2048

# The wiki's own family/baby Category: tags (via akinator_mode.py's SUPERCATEGORIES) that count
# as a genuinely documented evolution/variant/family relationship. "Baby or Young Character"
# covers Evolution Mode's own baby-stage signal; the rest are named family groups pulled
# directly from real wiki Category: tags.
FAMILY_TAGS = {
    "Baby or Young Character",
    "Sahur Family",
    "Signal Family",
    "Bombardiro or Bomber Family",
    "Crocodile Family",
    "67 Family",
    "Matteo Family",
    "Wolf Group",
}

# A page shorter than this is treated as too stub-like to generate solid stats/types/moves from
# later, even if it happens to carry a qualifying family tag.
MIN_CONTENT_LENGTH = 500
# Caps how many candidates any single family tag can contribute, so a large family (Sahur Family
# has ~130 members) can't crowd out every other theme in the pool the model sees.
MAX_PER_FAMILY_TAG = 8
# Overall cap on how many candidates get sent to the one Gemini call, keeping the prompt compact.
MAX_TOTAL_CANDIDATES = 40
# Each candidate's wiki content is truncated to this many characters in the prompt — enough for
# the model to judge lore richness and the actual relationship, without sending full pages for
# up to 40 candidates at once.
EXCERPT_LENGTH = 300
# Safety cap on the final shortlist size, mirroring the prompt's own 5-10 request.
MAX_SHORTLIST_SIZE = 10


class StarterCandidate(BaseModel):
    title: str  # must be copied verbatim from the candidate list
    relationship: str  # what the documented evolution/variant/family relationship actually is
    # A loose, informal type/vibe descriptor (e.g. "Fire-ish") for grouping in the shortlist —
    # deliberately NOT validated against rpg_types.POKEMON_TYPES; RPG Mode's own wiki-grounded
    # call is what does that properly, per character, and running it here for every candidate
    # would multiply this mode's Gemini call count for a "loose" grouping that doesn't need it.
    theme: str = ""
    reason: str = ""  # one-line reason it'd make a good starter


class StarterShortlist(BaseModel):
    picks: list[StarterCandidate] = []
    could_not_determine: bool = False
    # Required whenever the shortlist is empty, or fewer than 5 candidates were included despite
    # a larger candidate pool being available — never pad the list to hit a round number instead.
    undetermined_reason: str = ""


SYSTEM_INSTRUCTION = (
    "You are a 'starter' scout for the Italian Brainrot universe, in the style of choosing "
    "starter Pokémon. You are given a list of REAL candidate characters, each already confirmed "
    "by the wiki's own category tags to have a documented evolution, variant, or family "
    "relationship — your job is to pick the 5 to 10 BEST of them to present as starter options. "
    "You NEVER invent a candidate or a relationship: every pick's title must be copied verbatim "
    "from the candidate list, and its 'relationship' field must describe only what the given "
    "wiki excerpt actually states or clearly implies — never something you're guessing at.\n\n"
    "Prioritize candidates that are well-documented enough to build a solid RPG Mode Dex Entry "
    "from (clear appearance, personality, or combat lore) over stub-like entries with barely any "
    "real detail, even if a stub technically qualifies by its tag.\n\n"
    "Give each pick a short 'theme' label — a loose, informal type/vibe descriptor (e.g. "
    "'Fire-ish', 'Water-ish', 'Electric-ish') based on its apparent theme. This is not a strict "
    "system, just enough to help the user visually group a balanced trio. Also give a one-line "
    "'reason' explaining why it'd make a good starter, grounded in its actual lore/theme "
    "clarity.\n\n"
    "If the user requested a specific type/theme filter, prioritize candidates matching it — if "
    "genuinely none of the given candidates fit that filter well, say so honestly in "
    "undetermined_reason instead of forcing a weak match. If fewer than 5 of the given "
    "candidates are actually well-documented enough to recommend, return only however many "
    "genuinely qualify and explain why in undetermined_reason — never pad the list with weak or "
    "barely-documented entries just to reach a round number."
)
SYSTEM_INSTRUCTION = with_content_policy(SYSTEM_INSTRUCTION)


def _qualifying_candidates(content_by_title):
    """Code-only, free candidate narrowing — see the module docstring. Deterministic given the
    same wiki_data.json and tag data (dict iteration is insertion-ordered), so results are
    reproducible run to run rather than randomly varying."""
    picked_per_tag = {}
    seen = set()
    candidates = []
    for title, tags in akinator_mode.TAGS_BY_TITLE.items():
        matching_tags = tags & FAMILY_TAGS
        if not matching_tags or title in seen:
            continue
        if len(content_by_title.get(title, "")) < MIN_CONTENT_LENGTH:
            continue
        if not any(picked_per_tag.get(tag, 0) < MAX_PER_FAMILY_TAG for tag in matching_tags):
            continue
        seen.add(title)
        for tag in matching_tags:
            picked_per_tag[tag] = picked_per_tag.get(tag, 0) + 1
        candidates.append(title)
        if len(candidates) >= MAX_TOTAL_CANDIDATES:
            break
    return candidates


def _load_qualifying_content():
    """One-off full read of wiki_data.json, kept only for titles carrying a qualifying family
    tag (a few hundred out of ~4700) rather than held permanently — same one-off-read-per-request
    philosophy as akinator_mode.py's own _content_for_titles, just scaled to a larger but still
    bounded set."""
    qualifying_titles = {title for title, tags in akinator_mode.TAGS_BY_TITLE.items() if tags & FAMILY_TAGS}
    with open(akinator_mode.WIKI_DATA_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    return {page["title"]: page["content"] for page in pages if page["title"] in qualifying_titles}


def _build_prompt(filter_text, candidates_with_excerpts):
    filter_note = (
        f"The user asked for this specific type/theme filter: {filter_text}\n\n"
        if filter_text
        else "No specific type/theme filter was requested — give a well-rounded, varied shortlist.\n\n"
    )
    candidates_block = "\n\n---\n\n".join(f"[{title}]\n{excerpt}" for title, excerpt in candidates_with_excerpts)
    return (
        f"{filter_note}"
        f"Candidate real wiki characters, each already confirmed to have a documented "
        f"evolution/variant/family relationship via the wiki's own category tags (choose ONLY "
        f"from this list, copy titles exactly as written):\n\n{candidates_block}\n\n"
        f"Pick the 5-10 best starter candidates per your instructions."
    )


def generate_shortlist(gemini_client, filter_text, candidates_with_excerpts):
    response = call_with_retry(lambda: gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(filter_text, candidates_with_excerpts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=StarterShortlist,
        ),
    ))
    return response.parsed


def _verify_picks(picks, candidate_titles):
    """Defense in depth: even though the model is instructed to only copy from the candidate
    list, verify every returned title actually matches a real candidate (case-insensitive)
    before trusting it — same pattern evolution_mode.py's _verify_stages uses."""
    lookup = {title.strip().lower(): title for title in candidate_titles}
    verified = []
    seen = set()
    for pick in picks:
        real_title = lookup.get(pick.title.strip().lower())
        if not real_title or real_title in seen:
            continue
        seen.add(real_title)
        relationship = (pick.relationship or "").strip() or "a documented family/variant relationship"
        if relationship and relationship[-1] not in ".!?":
            relationship += "."
        reason = (pick.reason or "").strip() or "well-documented lore."
        verified.append(StarterCandidate(
            title=real_title,
            relationship=relationship,
            theme=(pick.theme or "").strip() or "Unthemed",
            reason=reason,
        ))
    return verified[:MAX_SHORTLIST_SIZE]


def format_shortlist(picks, undetermined_reason):
    if not picks:
        reason = undetermined_reason.strip() or "not enough well-documented candidates with a real evolution/variant/family relationship were found."
        return f"Couldn't put together a starter shortlist — {reason}"

    lines = ["Starter candidates (grouped loosely by theme):"]
    for pick in picks:
        lines.append(f"- {pick.title} [{pick.theme}] — {pick.relationship} {pick.reason}")
    if undetermined_reason.strip():
        lines.append("")
        lines.append(f"Note: {undetermined_reason.strip()}")
    return "\n".join(lines)


def build_starter_shortlist(gemini_client, filter_text):
    content_by_title = _load_qualifying_content()
    candidate_titles = _qualifying_candidates(content_by_title)
    if not candidate_titles:
        return "No well-documented characters with a real evolution/variant/family relationship were found."

    candidates_with_excerpts = [(title, content_by_title[title][:EXCERPT_LENGTH]) for title in candidate_titles]

    shortlist = generate_shortlist(gemini_client, filter_text, candidates_with_excerpts)
    if shortlist is None:
        return "Couldn't generate a starter shortlist — try asking again."

    picks = _verify_picks(shortlist.picks, candidate_titles)
    return format_shortlist(picks, shortlist.undetermined_reason)
