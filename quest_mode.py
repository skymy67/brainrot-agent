#!/usr/bin/env python3
"""Quest Mode: a continuing, chaptered Pokémon-style Brainrot adventure campaign.

Unlike the other one-shot modes, Quest Mode is stateful and round-trips its state through the
client exactly like Akinator Mode does (no server-side session store). Each turn writes one new
chapter continuing the same campaign, using only real wiki Brainrots the player hasn't already
met as new encounters/quest-givers/bosses — tracked in `seen_titles` — so a long playthrough
genuinely works its way through the wiki's full ~4687-character cast instead of reusing the same
handful every time.

The full story text is never resent to Gemini turn over turn (it would grow without bound and
eventually blow past the context/token budget); each call instead asks for a compact
`updated_summary` of the whole campaign so far, and that summary is what continuity is built
from on the next turn.
"""

import random

from google.genai import types
from pydantic import BaseModel

import akinator_mode  # reuses its already-loaded ALL_TITLES as a fallback candidate pool
from content_policy import with_content_policy
from gemini_retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 768
MAX_OUTPUT_TOKENS = 2048
# Requested from RAG per turn — only a fraction end up both unique and actually unseen, so this
# is deliberately generous relative to how many candidates a chapter will realistically use.
CANDIDATES_PER_CHAPTER = 40
MIN_UNSEEN_CANDIDATES = 6
FALLBACK_SAMPLE_SIZE = 10


class QuestState(BaseModel):
    chapter_count: int = 0
    goal: str = ""  # the player's original adventure brief, kept stable across turns
    seen_titles: list[str] = []
    summary: str = ""


class QuestChapterOutput(BaseModel):
    chapter_title: str
    chapter_text: str
    updated_summary: str
    # Real candidate titles (copied verbatim) actually featured as named characters this
    # chapter — verified against the candidate list before being trusted, same defense-in-depth
    # pattern evolution_mode.py uses for its own model-picked titles.
    featured_titles: list[str] = []


SYSTEM_INSTRUCTION = (
    "You are a Pokémon-style RPG game master running a continuing campaign in the Italian "
    "Brainrot universe for one player. Each turn you write exactly ONE new chapter that "
    "continues directly from the campaign summary you're given — never restart or contradict "
    "it. Feature real Brainrot characters as wild encounters, quest-givers, or bosses, choosing "
    "ONLY from the candidate list provided each turn (copy titles into featured_titles exactly "
    "as written); never invent a Brainrot name that isn't in the candidate list. Stay accurate "
    "to each featured Brainrot's real lore, abilities, and personality, but freely invent the "
    "plot, locations, and dialogue connecting them. Keep updated_summary a tight 3-5 sentence "
    "recap of the WHOLE campaign so far (not just this chapter) so future chapters can pick up "
    "the thread without re-reading full chapter text."
)
SYSTEM_INSTRUCTION = with_content_policy(SYSTEM_INSTRUCTION)


def _dedupe_titles(documents, metadatas):
    seen, titles, docs_by_title = set(), [], {}
    for doc, meta in zip(documents, metadatas):
        title = meta["title"]
        if title not in seen:
            seen.add(title)
            titles.append(title)
            docs_by_title[title] = doc
    return titles, docs_by_title


def _unseen_candidates(documents, metadatas, seen_titles):
    """Deterministically excludes anything already featured — the model can only pick a Brainrot
    it genuinely hasn't used yet, so "no repeats" is enforced in code, not just by instruction."""
    titles, docs_by_title = _dedupe_titles(documents, metadatas)
    seen = set(seen_titles)
    unseen = [title for title in titles if title not in seen]

    if len(unseen) < MIN_UNSEEN_CANDIDATES:
        # This turn's RAG query mostly returned Brainrots already used — top up with a random
        # sample from the full wiki roster so a long-running campaign never stalls out for lack
        # of fresh candidates. These extras have no retrieved content chunk, just a name.
        pool = [title for title in akinator_mode.ALL_TITLES if title not in seen and title not in unseen]
        extra = random.sample(pool, min(FALLBACK_SAMPLE_SIZE, len(pool)))
        for title in extra:
            docs_by_title.setdefault(title, "(no wiki excerpt retrieved for this chapter — use the name only)")
        unseen = unseen + extra

    return unseen, docs_by_title


def _format_candidates(titles, docs_by_title):
    return "\n\n---\n\n".join(f"[{title}]\n{docs_by_title[title]}" for title in titles)


def _call_gemini(gemini_client, prompt):
    response = call_with_retry(lambda: gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=QuestChapterOutput,
        ),
    ))
    return response.parsed


def process_turn(gemini_client, state_dict, player_input, documents, metadatas):
    """Entry point. state_dict is the previous turn's returned state (None to start a new
    campaign). player_input is the player's free-text message: the adventure brief on the first
    turn, or their next action/steer on later turns (may be blank to just continue). documents/
    metadatas are this turn's RAG retrieval results — app.py decides the query (the original
    goal on later turns, so retrieval stays anchored to the campaign's own theme). Returns
    (chapter_message, new_state_dict)."""
    try:
        state = QuestState(**state_dict) if state_dict else None
    except (TypeError, ValueError):
        state = None

    player_input = (player_input or "").strip()

    if state is None:
        goal = player_input or "A Brainrot adventure."
        candidates, docs_by_title = _unseen_candidates(documents, metadatas, [])
        prompt = (
            f"Start a brand-new campaign. The player's request for the adventure: {goal}\n\n"
            f"Candidate real Brainrot characters for this chapter:\n"
            f"{_format_candidates(candidates, docs_by_title)}\n\n"
            "Write Chapter 1."
        )
    else:
        goal = state.goal
        candidates, docs_by_title = _unseen_candidates(documents, metadatas, state.seen_titles)
        action = player_input or "Continue the story."
        prompt = (
            f"Campaign so far: {state.summary}\n\n"
            f"Chapters completed: {state.chapter_count}\n\n"
            f"Player's next move: {action}\n\n"
            "Candidate real Brainrot characters for this chapter (none of these have appeared "
            f"in the campaign yet):\n{_format_candidates(candidates, docs_by_title)}\n\n"
            f"Write Chapter {state.chapter_count + 1}, continuing directly from the summary above."
        )

    parsed = _call_gemini(gemini_client, prompt)
    featured = list(dict.fromkeys(title for title in parsed.featured_titles if title in candidates))

    prior_seen = state.seen_titles if state else []
    new_state = QuestState(
        chapter_count=(state.chapter_count if state else 0) + 1,
        goal=goal,
        seen_titles=prior_seen + featured,
        summary=parsed.updated_summary,
    )
    chapter_message = f"**Chapter {new_state.chapter_count}: {parsed.chapter_title}**\n\n{parsed.chapter_text}"
    return chapter_message, new_state.model_dump()
