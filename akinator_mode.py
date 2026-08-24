#!/usr/bin/env python3
"""Akinator Mode: a 20-questions-style guessing game restricted to real wiki characters.

Every question topic and every guess is chosen from the actual set of characters in
wiki_data.json — the model is never asked to invent one. To make that tractable without
feeding thousands of full wiki pages to Gemini on every turn, the candidate-narrowing itself
is a deterministic decision-tree over each page's own MediaWiki `Category:` tags (mined once
at startup, the same first-party signal already validated for Evolution Mode): at each turn we
pick the tag that splits the current candidate pool closest to 50/50, which is the trait that
carries the most information about which half of the pool the answer falls into. Gemini's role
is narrower than "solve the game" — it phrases that chosen trait into a natural question, and
once the pool has narrowed to a small shortlist, picks the best-fitting final guess from it.

Session state (candidate pool, asked tags, question count, round history) has nowhere to live
server-side in this app — there's no database or session store, and everything else the
backend does is a single stateless request/response. So state round-trips through the
request/response bodies each turn, the same way the frontend already persists chat history to
sessionStorage; the caller (app.py) is only responsible for handing back whatever dict this
module last returned.
"""

import json
import re
from typing import Literal

from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

WIKI_DATA_FILE = "wiki_data.json"
GEMINI_MODEL = "gemini-3.6-flash"
MAX_QUESTIONS = 30
MAX_SHORTLIST_FOR_GEMINI = 8
CONTENT_TRUNCATE_CHARS = 500
QUESTION_THINKING_BUDGET = 256
QUESTION_MAX_OUTPUT_TOKENS = 256
GUESS_THINKING_BUDGET = 512
GUESS_MAX_OUTPUT_TOKENS = 512

# Category tags can be multi-word ("Category:Not Italian", "Category:Sahur family"), and on
# ~31 of the wiki's 4687 pages, multiple categories appear back-to-back on the same source line
# with no newline between them at all (a mwparserfromhell plain-text rendering quirk). Matching
# only up to \n (or only up to the next literal newline) either truncates multi-word tags to
# their first word, or — worse — swallows every subsequent "Category:X" on that line into one
# giant garbage "tag" (e.g. "Category:Ostrich Category:Toilet" with no separator became the
# single fake tag "OstrichCategory:ToiletCategory:Train"). That silently dropped real tags for
# those pages, degrading how well the pool actually narrows for them. Stopping the match at the
# next "Category:" occurrence too (not just \n) splits these correctly.
CATEGORY_RE = re.compile(r"Category:(.*?)(?=Category:|\n|$)")

# Tags that exist on real wiki pages but aren't player-observable traits of the character
# itself — production/attribution metadata (which AI tool or human made the page), wiki
# housekeeping/editorial status, release-date or meme-age tracking, or anything tier/power/
# popularity-flavored (explicitly out of scope per the spec: rarity-tier and stat-like data
# isn't fair question material here, RPG/Rarity mode already cover that). A player thinking of
# a character has no way to know these about their character's wiki page, so a tag like this
# getting picked as a question forces an "unsure" answer that narrows nothing and wastes a turn.
# Identified by inspecting the full ~900-tag frequency table (not just the top ones) for
# editorial/attribution/meta patterns; an obscure one attached to only a page or two may still
# slip through uncaught, an acceptable, disclosed limitation rather than a correctness bug.
EXCLUDED_TAGS = {
    "Wikimades",
    "Wikimade",
    "Legacy DALL-E",
    "Gemini",
    "Non-AI",
    "Characters",
    "Italian Brainrot Characters",
    "Famous",
    "Popular in Roblox",
    "Strongest",
    "Alexey Pigeon",
    "Noxa",
    "Articles with potentially offensive material",
    "Italian brainrot creators",
    "AI Reviewed Pages",
    "Candidates for Shining Articles",
    "Shining article",
    "Featured",
    "Stubs",
    "Long articles",
    "Disambiguation",
    "Non-Mainstream Content",
    "Non-mainstream content",
    "Unverified content",
    "Uncategorized Pages",
    "Formerly Deleted",
    "Deleted",
    "Old Wiki",
    "Undeleted pages",
    "Steal a Brainrot-originating",
    "Craft a Brainrot Exclusive",
    "Very Young Brainrots",
    "Non-existent Brainrot",
    "Non-Brainrots",
    "Bing DALL-E",
    "Dreamina AI",
    # In-game/collector rarity tiers — same "Steal a Brainrot" tier system Rarity Mode already
    # covers, explicitly out of scope per the spec alongside the numeric "Tier X-Y" tags below.
    "Common",
    "Uncommon",
    "Rare",
    "Epic",
    "Legendary",
    "Mythic",
    "Brainrot God",
    "Secret",
    "Celestial",
    "OG",
}
EXCLUDED_TAG_PATTERNS = [
    re.compile(r"^Tier\b", re.IGNORECASE),  # power-tier classifications, e.g. "Tier 10-C"
    re.compile(r"'s [Bb]rainrots$"),  # creator-attribution tags, e.g. "Henzwxz's Brainrots"
    re.compile(r"^Brainrots of \w+ \d{4}$"),  # release-period tags, e.g. "Brainrots of March 2025"
    re.compile(r"^\w+ \d{4} Brainrots$"),  # release-period tags, other word order
    re.compile(r"\d Years?.? Old Brainrots$", re.IGNORECASE),  # meme-age tags
    re.compile(r"\bAi Video Maker\b", re.IGNORECASE),  # AI-generation-tool attribution
]


def _is_excluded_tag(tag):
    if tag in EXCLUDED_TAGS:
        return True
    return any(pattern.search(tag) for pattern in EXCLUDED_TAG_PATTERNS)


def _load_characters():
    """Only titles and their (small) tag sets are kept in memory for the process lifetime —
    holding every page's full content in a second permanent copy (on top of what build_index.py
    and chroma_db already need in memory to build/serve the RAG index) contributed to an
    out-of-memory crash on Railway's production instance. Full content is only ever needed for
    a handful of shortlisted candidates at final-guess time, so it's loaded lazily via
    _content_for_titles() instead of held permanently."""
    with open(WIKI_DATA_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    titles = [page["title"] for page in pages]
    tags_by_title = {
        page["title"]: {tag.strip() for tag in CATEGORY_RE.findall(page["content"]) if not _is_excluded_tag(tag.strip())}
        for page in pages
    }
    return titles, tags_by_title


ALL_TITLES, TAGS_BY_TITLE = _load_characters()


def _content_for_titles(titles):
    """Re-reads wiki_data.json for just the given titles' content. Called only when making a
    final guess among a small shortlist (at most MAX_SHORTLIST_FOR_GEMINI candidates), never on
    the hot path of narrowing the pool — a full JSON parse is an acceptable one-off cost there,
    much cheaper over the life of the process than holding every page's content in RAM always."""
    wanted = set(titles)
    with open(WIKI_DATA_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    return {page["title"]: page["content"] for page in pages if page["title"] in wanted}


# --- State ---------------------------------------------------------------------------------


class AkinatorState(BaseModel):
    phase: Literal["asking", "guessing"] = "asking"
    candidate_titles: list[str]
    asked_tags: list[str] = []
    question_count: int = 0
    pending_tag: str | None = None
    pending_guess: str | None = None
    final_guess_made: bool = False
    history: list[str] = []


# --- Deterministic candidate narrowing ------------------------------------------------------


def _best_split_tag(candidate_titles, asked_tags):
    """The tag whose presence among the current pool is closest to a 50/50 split carries the
    most information about which half the answer will fall into — the same greedy heuristic a
    real 20-questions solver uses. Already-asked tags are excluded so we never repeat a
    question. Returns None once no remaining tag usefully splits the pool (e.g. every
    remaining candidate happens to share the same tags, or none are tagged at all)."""
    asked = set(asked_tags)
    tag_counts = {}
    for title in candidate_titles:
        for tag in TAGS_BY_TITLE.get(title, ()):
            if tag in asked:
                continue
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # A tag held by every candidate (or none) carries zero information — skip it.
    useful = {tag: count for tag, count in tag_counts.items() if 0 < count < len(candidate_titles)}
    if not useful:
        return None

    half = len(candidate_titles) / 2
    return min(useful, key=lambda tag: abs(useful[tag] - half))


def _filter_pool(candidate_titles, tag, answer):
    if answer == "unsure" or tag is None:
        return list(candidate_titles)
    has_tag = lambda title: tag in TAGS_BY_TITLE.get(title, ())
    if answer == "yes":
        return [title for title in candidate_titles if has_tag(title)]
    return [title for title in candidate_titles if not has_tag(title)]


# --- Gemini calls (with deterministic fallbacks if they fail) ------------------------------


class QuestionPhrasing(BaseModel):
    question_text: str


QUESTION_SYSTEM_INSTRUCTION = (
    "You are the host of a 20-questions-style guessing game about Italian Brainrot wiki "
    "characters. You are given ONE trait (a wiki category tag) that the game engine has "
    "already chosen as the most useful thing to ask about next — your only job is to phrase "
    "it as one short, natural yes/no question a player can answer at a glance. Never mention "
    "'category' or 'tag' — phrase it as a normal trait question about appearance, species/base "
    "object, or a similar observable trait. Keep it under 20 words. Never introduce a trait "
    "other than the one given, and never reference or guess a specific character name."
)


def _fallback_question_text(tag):
    return f"Does your character have anything to do with '{tag}' (e.g. {tag.lower()}-themed or {tag.lower()}-related)?"


def _phrase_question(gemini_client, tag, history_lines):
    history_summary = "\n".join(history_lines[-10:]) or "(none yet)"
    prompt = (
        f"Trait to ask about: {tag}\n\n"
        f"Questions already asked this round (for phrasing variety, don't repeat wording):\n"
        f"{history_summary}\n\n"
        f"Write one short yes/no question asking whether the player's character has this trait."
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=QUESTION_SYSTEM_INSTRUCTION,
                max_output_tokens=QUESTION_MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=QUESTION_THINKING_BUDGET),
                response_mime_type="application/json",
                response_schema=QuestionPhrasing,
            ),
        )
    except genai_errors.APIError:
        return _fallback_question_text(tag)

    parsed = response.parsed
    if parsed is None or not parsed.question_text.strip():
        return _fallback_question_text(tag)
    return parsed.question_text.strip()


class GuessPick(BaseModel):
    guess_title: str


GUESS_SYSTEM_INSTRUCTION = (
    "You are the host of a 20-questions-style guessing game about Italian Brainrot wiki "
    "characters, making a final guess from a short list of real candidates. You must pick "
    "EXACTLY one title, copied verbatim from the candidate list given — never invent or "
    "modify a name, and never pick anything not in that list."
)


def _pick_best_guess(gemini_client, shortlist_titles, history_lines):
    history_summary = "\n".join(history_lines[-10:]) or "(no questions asked yet)"
    content_by_title = _content_for_titles(shortlist_titles)
    context = "\n\n---\n\n".join(
        f"[{title}]\n{content_by_title.get(title, '')[:CONTENT_TRUNCATE_CHARS]}" for title in shortlist_titles
    )
    prompt = (
        f"Candidates (choose ONLY one of these, copied exactly):\n"
        + "\n".join(f"- {title}" for title in shortlist_titles)
        + f"\n\nWiki context on each candidate:\n{context}\n\n"
        f"Questions asked this round and the player's answers:\n{history_summary}\n\n"
        f"Pick the single candidate that best matches all the answers given."
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GUESS_SYSTEM_INSTRUCTION,
                max_output_tokens=GUESS_MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=GUESS_THINKING_BUDGET),
                response_mime_type="application/json",
                response_schema=GuessPick,
            ),
        )
    except genai_errors.APIError:
        return shortlist_titles[0]

    parsed = response.parsed
    if parsed is None:
        return shortlist_titles[0]

    lookup = {title.strip().lower(): title for title in shortlist_titles}
    return lookup.get(parsed.guess_title.strip().lower(), shortlist_titles[0])


# --- Turn orchestration ----------------------------------------------------------------------


def _make_guess(gemini_client, pool, asked_tags, question_count, history, final_guess_made):
    shortlist = pool[:MAX_SHORTLIST_FOR_GEMINI]
    guess = shortlist[0] if len(shortlist) == 1 else _pick_best_guess(gemini_client, shortlist, history)
    new_state = AkinatorState(
        phase="guessing",
        candidate_titles=pool,
        asked_tags=asked_tags,
        question_count=question_count,
        pending_guess=guess,
        final_guess_made=final_guess_made,
        history=history,
    )
    return f"Is it **{guess}**?", new_state


def _advance(gemini_client, pool, asked_tags, question_count, history, final_guess_made):
    if not pool:
        return (
            "None of the real wiki characters fit all those answers — I'm stuck! Want to tell "
            "me who it was, or start a new round?"
        ), None

    if len(pool) == 1:
        return _make_guess(gemini_client, pool, asked_tags, question_count, history, final_guess_made)

    if question_count >= MAX_QUESTIONS:
        return _make_guess(gemini_client, pool, asked_tags, question_count, history, final_guess_made=True)

    tag = _best_split_tag(pool, asked_tags)
    if tag is None:
        return _make_guess(gemini_client, pool, asked_tags, question_count, history, final_guess_made)

    question_text = _phrase_question(gemini_client, tag, history)
    new_state = AkinatorState(
        phase="asking",
        candidate_titles=pool,
        asked_tags=asked_tags + [tag],
        question_count=question_count,
        pending_tag=tag,
        final_guess_made=final_guess_made,
        history=history,
    )
    return question_text, new_state


def _handle_question_answer(gemini_client, state, answer):
    pool = _filter_pool(state.candidate_titles, state.pending_tag, answer)
    question_count = state.question_count + 1
    history = state.history + [f"Q: {state.pending_tag}? A: {answer}"]
    return _advance(gemini_client, pool, state.asked_tags, question_count, history, state.final_guess_made)


def _handle_guess_answer(gemini_client, state, answer):
    history = state.history + [f"Guess: {state.pending_guess}? A: {answer}"]
    if answer == "yes":
        return f"Got it — it was **{state.pending_guess}**! 🎉", None

    if state.final_guess_made:
        return (
            f"You've stumped me! {MAX_QUESTIONS} questions and my final guess weren't enough. "
            f"Want to tell me who it was?"
        ), None

    pool = [title for title in state.candidate_titles if title != state.pending_guess]
    return _advance(gemini_client, pool, state.asked_tags, state.question_count, history, state.final_guess_made)


def process_turn(gemini_client, state_dict, answer):
    """Entry point. state_dict is the previous turn's returned state (None to start a new
    round), answer is "yes" | "no" | "unsure" | "reset" from the player's last button press.
    Returns (message_text, new_state_dict_or_None) — None means the round has ended."""
    answer = (answer or "").strip().lower()

    if answer == "reset":
        return "Round ended — start a new one whenever you're ready!", None

    if state_dict is not None:
        try:
            state = AkinatorState(**state_dict)
        except (TypeError, ValueError):
            state = None
    else:
        state = None

    if state is None:
        message, new_state = _advance(gemini_client, list(ALL_TITLES), [], 0, [], False)
        return message, (new_state.model_dump() if new_state else None)

    if answer not in ("yes", "no", "unsure"):
        answer = "unsure"

    if state.phase == "asking":
        message, new_state = _handle_question_answer(gemini_client, state, answer)
    else:
        message, new_state = _handle_guess_answer(gemini_client, state, answer)

    return message, (new_state.model_dump() if new_state else None)
