#!/usr/bin/env python3
"""Evolution Mode: builds a Pokémon-style evolution line using only real wiki characters.

Every stage other than the target itself must be a real wiki page. The model is only allowed
to pick from an explicit candidate list drawn from the wiki's own RAG retrieval (it is never
asked to free-generate a name), and every title it returns is verified against that list in
code before being used. If nothing real fits, the mode says the character has no evolution
line instead of inventing or loosely matching a name to fill a slot.
"""

from typing import Literal

from google.genai import types
from pydantic import BaseModel

import rarity_mode
from content_policy import with_content_policy
from gemini_retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 640
MAX_OUTPUT_TOKENS = 1024
MAX_STAGES_PER_SIDE = 2

HIGH_TIER_RARITIES = {"Legendary", "Mythic", "Brainrot God", "Secret"}


class EvolutionSelection(BaseModel):
    target_position: Literal["stage1", "middle", "final"]
    prevolutions: list[str] = []  # earliest-to-latest, must be copied verbatim from the candidate list
    evolutions: list[str] = []  # earliest-to-latest, must be copied verbatim from the candidate list


SYSTEM_INSTRUCTION = (
    "You are an evolution-line designer for the Italian Brainrot universe, in the style of a "
    "Pokémon-style evolution chain. You NEVER invent or free-generate a character name — every "
    "stage other than the target character itself must be copied EXACTLY from the CANDIDATE "
    "list given in the prompt. First lock in the TARGET character's own position in the line "
    "(stage1 / middle / final) using only its own name, lore, and tier. Then, from the "
    "candidate list only, select the real character(s) — ordered earliest to latest — that "
    "plausibly precede and/or follow it. A real match requires BOTH a name/theme link AND the "
    "same underlying creature/object base as the target, just at a different life stage or "
    "size — a shared word or theme alone is not enough if the candidate is actually a "
    "different kind of thing (e.g. a rat character and an unrelated cheese-block character are "
    "not the same base just because one is themed around the other's food).\n\n"
    "A candidate's NAME PREFIX ALONE ('Bambino [Name]', 'Los [Name]itos', 'Las [Name]itas', "
    "etc.) is NOT a reliable signal by itself — some 'Bambino'-named pages turn out to be "
    "unrelated spinoff characters, and some 'Los'/'Las'-named pages turn out to be genuine baby "
    "versions of the target despite being worded as its 'children' in their own summary. Judge "
    "each candidate on its actual content instead:\n"
    "- Look for this wiki's own 'Category:Baby' or 'Category:Bambino' tag, usually listed near "
    "the end of a page's content — this is the wiki's own first-party signal that a page is a "
    "genuine baby/early-life-stage version of another character, and should be weighed heavily "
    "even if the page's summary text uses looser family language like 'children of X'.\n"
    "- Also look for explicit prose describing the candidate as a baby/young version of the "
    "target (e.g. a History section noting it originated as 'baby versions of [target]').\n"
    "- A candidate with NEITHER a Baby/Bambino category NOR any baby/young-version language, "
    "described only as a cousin, unrelated spinoff, or separate meme with no stated tie to the "
    "target's own life stages, is weaker evidence and should generally be treated as not a "
    "valid stage.\n\n"
    "If locked as stage1, only select from evolutions (things that come after); if locked as "
    "final, only select prevolutions (things that come before); if locked as middle, select at "
    "most one of each. Select at most two stages on any one side. If no candidate plausibly "
    "fits a side, leave that list empty rather than forcing a weak match — never pick a "
    "candidate just to fill a slot, and never reject a genuinely valid baby-stage match just "
    "because a weaker or unrelated candidate also happens to be in the list."
)
SYSTEM_INSTRUCTION = with_content_policy(SYSTEM_INSTRUCTION)


def _format_candidates(candidate_titles):
    return "\n".join(f"- {title}" for title in candidate_titles)


def _build_prompt(character_name, wiki_context, candidate_titles):
    return (
        f"Context from the Italian Brainrot wiki:\n\n{wiki_context}\n\n"
        f"Target character: {character_name}\n\n"
        f"Candidate real wiki characters (choose ONLY from this list, copy titles exactly as "
        f"written, do not modify or invent any):\n{_format_candidates(candidate_titles)}\n\n"
        "Determine the target's own position, then select real prevolution/evolution "
        "candidates from the list above per your instructions."
    )


def select_real_stages(gemini_client, character_name, wiki_context, candidate_titles):
    response = call_with_retry(lambda: gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(character_name, wiki_context, candidate_titles),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=EvolutionSelection,
        ),
    ))
    return response.parsed


def _verify_titles(titles, candidate_titles):
    """Defense in depth: even though the model is instructed to only copy from the candidate
    list, verify every returned title actually matches a real candidate (case-insensitive)
    before trusting it. Anything that doesn't match exactly — a reworded name, a hallucination
    despite instructions — is dropped rather than used."""
    lookup = {title.strip().lower(): title for title in candidate_titles}
    verified = []
    for title in titles:
        match = lookup.get(title.strip().lower())
        if match and match not in verified:
            verified.append(match)
    return verified[:MAX_STAGES_PER_SIDE]


def _assign_levels(stage_count, known_tier=None):
    nudge = 4 if known_tier in HIGH_TIER_RARITIES else 0
    levels = [1]
    if stage_count >= 2:
        levels.append(14 + nudge)
    if stage_count >= 3:
        levels.append(30 + nudge)
    return levels


def _format_line(stage_names, known_tier):
    levels = _assign_levels(len(stage_names), known_tier)
    return "\n".join(f"Stage {i + 1}: {name} — Lv {level}" for i, (name, level) in enumerate(zip(stage_names, levels)))


def build_evolution_line(gemini_client, character_name, wiki_context, candidate_titles):
    if not candidate_titles:
        return f"{character_name} has no evolution line."

    selection = select_real_stages(gemini_client, character_name, wiki_context, candidate_titles)
    if selection is None:
        return f"{character_name} has no evolution line."

    prevolutions, evolutions = [], []
    if selection.target_position == "stage1":
        evolutions = _verify_titles(selection.evolutions, candidate_titles)
    elif selection.target_position == "final":
        prevolutions = _verify_titles(selection.prevolutions, candidate_titles)
    else:
        prevolutions = _verify_titles(selection.prevolutions, candidate_titles)[:1]
        evolutions = _verify_titles(selection.evolutions, candidate_titles)[:1]

    stage_names = prevolutions + [character_name] + evolutions
    if len(stage_names) < 2:
        return f"{character_name} has no evolution line."

    known_tier = rarity_mode.check_known_game_rarity(character_name)
    return _format_line(stage_names, known_tier)
