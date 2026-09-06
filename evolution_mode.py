#!/usr/bin/env python3
"""Evolution Mode: builds a Pokémon-style evolution line using only real wiki characters.

Every stage other than the target itself must be a real wiki page. The model is only allowed
to pick from an explicit candidate list drawn from the wiki's own RAG retrieval (it is never
asked to free-generate a name), and every title it returns is verified against that list in
code before being used. If nothing real fits, the mode says the character has no evolution
line instead of inventing or loosely matching a name to fill a slot.

Each stage transition also gets an evolution METHOD, not just a level. 'level' is the default,
unchanged from before (the existing tier-nudged 1 / 14 / 30 progression). The model may instead
pick 'stone' (grounded in the arriving candidate's own established elemental/thematic identity —
using the real Pokémon evolution stone names, not invented ones), 'trade' (grounded in a
documented duo/pairing/exchange relationship with another character), or 'friendship' (grounded
in documented loyalty/bonding/being-raised-by-someone) — always requiring a one-line citation of
the specific wiki detail behind the pick, exactly like Signature Move's is_generic/basis in
rpg_mode.py. No evidence, no non-level method — same "never invent" rule as everything else
here. A stone/trade/friendship stage shows that method instead of a level number, matching how
none of those three use a level requirement in real Pokémon either.

Each StageSelection's method describes how THAT stage is reached from whatever immediately
precedes it in the final chain — uniform in both directions (prevolutions and evolutions alike).
Since the target character itself has no list entry of its own, its arrival method (relevant
only when it actually has a prevolution to arrive from) is carried separately on
EvolutionSelection.target_method/target_stone/target_reasoning. The very first stage in the
whole chain never has a real arrival method (nothing precedes it) and always renders as the
plain starting level, regardless of anything selected for it.
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

EVOLUTION_METHODS = ["level", "stone", "trade", "friendship"]
FALLBACK_METHOD = "level"  # the safe default whenever a claimed method can't be trusted

# The real Pokémon evolution stones, reused end to end instead of inventing brainrot-flavored
# ones — same reasoning as rpg_types.POKEMON_TYPES: one shared, real vocabulary rather than a
# custom one, so a "stone" method's stone name is always something recognizable.
EVOLUTION_STONES = [
    "Fire Stone", "Water Stone", "Thunder Stone", "Leaf Stone", "Moon Stone",
    "Sun Stone", "Shiny Stone", "Dusk Stone", "Dawn Stone", "Ice Stone",
]


class StageSelection(BaseModel):
    title: str  # must be copied verbatim from the candidate list
    method: str = FALLBACK_METHOD  # one of EVOLUTION_METHODS — how THIS stage is reached
    stone: str = ""  # only meaningful when method == "stone" — one of EVOLUTION_STONES
    # Required whenever method != "level": the specific wiki detail behind the pick. Left "" for
    # "level", which needs no special justification since it's the default.
    reasoning: str = ""


class EvolutionSelection(BaseModel):
    target_position: Literal["stage1", "middle", "final"]
    prevolutions: list[StageSelection] = []  # earliest-to-latest, titles copied verbatim
    evolutions: list[StageSelection] = []  # earliest-to-latest, titles copied verbatim
    # How the TARGET character itself is reached — only meaningful (and only worth setting)
    # when target_position is "middle" or "final", i.e. the target actually has an immediate
    # prevolution to arrive from. Left at the defaults when target_position is "stage1".
    target_method: str = FALLBACK_METHOD
    target_stone: str = ""
    target_reasoning: str = ""


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
    "because a weaker or unrelated candidate also happens to be in the list.\n\n"
    "For every stage you select (each prevolution, each evolution, AND the target's own arrival "
    "via target_method/target_stone/target_reasoning when it has a prevolution), also decide "
    "the evolution method by which THAT stage is reached from whatever immediately precedes it "
    "in the line:\n"
    "- 'level' — the default. Use this unless a stage clearly qualifies for one of the others "
    "below.\n"
    "- 'stone' — use ONLY when the arriving stage's own wiki text clearly establishes a "
    "specific elemental/thematic identity (e.g. genuinely fire-themed, water-themed, ice-"
    "themed) that matches one of the real evolution stones given in the prompt. Set 'stone' to "
    "that single best-matching stone name.\n"
    "- 'trade' — use ONLY when the arriving stage's own wiki text documents a specific duo, "
    "pairing, or exchange relationship with another named character.\n"
    "- 'friendship' — use ONLY when the arriving stage's own wiki text documents loyalty, "
    "bonding, or being raised/cared for by someone.\n"
    "Whenever you pick anything other than 'level', you MUST cite the exact wiki detail behind "
    "that choice in the matching reasoning field — if you can't point to something concrete, "
    "use 'level' instead. Never pick stone/trade/friendship just to add variety; most stages "
    "should stay 'level', same as a typical Pokémon line where only some evolutions use a "
    "special method."
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
        f"Evolution stones (for the 'stone' method's stone field only): {', '.join(EVOLUTION_STONES)}\n\n"
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


def _verify_method_fields(method, stone, reasoning):
    """Defense in depth for one stage's evolution-method claim: a method outside
    EVOLUTION_METHODS falls back to plain level. A non-level method with no reasoning attached
    is treated as unsubstantiated and also falls back to level — an unbacked trade/friendship/
    stone claim is never kept just because the model asserted it. A 'stone' claim additionally
    needs its stone name to match the real vocabulary, or it falls back too (a stone method with
    an unrecognized stone is worse than just showing a level)."""
    method = (method or "").strip().lower()
    if method not in EVOLUTION_METHODS:
        method = FALLBACK_METHOD

    reasoning = (reasoning or "").strip()
    if method != "level" and not reasoning:
        method = FALLBACK_METHOD

    matched_stone = ""
    if method == "stone":
        stone_lookup = {s.lower(): s for s in EVOLUTION_STONES}
        matched_stone = stone_lookup.get((stone or "").strip().lower(), "")
        if not matched_stone:
            method = FALLBACK_METHOD

    if method == "level":
        reasoning = ""

    return method, matched_stone, reasoning


def _verify_stages(stages, candidate_titles):
    """Defense in depth: even though the model is instructed to only copy from the candidate
    list, verify every returned title actually matches a real candidate (case-insensitive)
    before trusting it — a reworded name, a hallucination despite instructions, is dropped
    rather than used. Each surviving stage's method/stone/reasoning is separately re-verified
    via _verify_method_fields."""
    lookup = {title.strip().lower(): title for title in candidate_titles}
    verified = []
    seen = set()
    for stage in stages:
        real_title = lookup.get(stage.title.strip().lower())
        if not real_title or real_title in seen:
            continue
        seen.add(real_title)
        method, stone, reasoning = _verify_method_fields(stage.method, stage.stone, stage.reasoning)
        verified.append(StageSelection(title=real_title, method=method, stone=stone, reasoning=reasoning))
    return verified[:MAX_STAGES_PER_SIDE]


def _assign_levels(stage_count, known_tier=None):
    nudge = 4 if known_tier in HIGH_TIER_RARITIES else 0
    levels = [1]
    if stage_count >= 2:
        levels.append(14 + nudge)
    if stage_count >= 3:
        levels.append(30 + nudge)
    return levels


def _stage_entries(prevolutions, target_name, target_method, target_stone, target_reasoning, evolutions):
    """Builds (name, method, stone, reasoning) tuples for the whole chain in order. Each
    prevolution/evolution's own method describes how IT is reached from whatever precedes it —
    uniform in both directions — and the target's own arrival (only relevant when it has a
    prevolution) is spliced in from the separate target_method/target_stone/target_reasoning
    fields, since the target has no list entry of its own. The very first stage in the whole
    chain never has a real arrival method (nothing precedes it), regardless of what was
    selected for it, so it's always forced back to a plain level."""
    entries = [(stage.title, stage.method, stage.stone, stage.reasoning) for stage in prevolutions]
    if prevolutions:
        entries.append((target_name, target_method, target_stone, target_reasoning))
    else:
        entries.append((target_name, "level", "", ""))
    entries.extend((stage.title, stage.method, stage.stone, stage.reasoning) for stage in evolutions)

    first_name, _, _, _ = entries[0]
    entries[0] = (first_name, "level", "", "")
    return entries


def _format_line(entries, known_tier):
    levels = _assign_levels(len(entries), known_tier)
    lines = []
    for i, ((name, method, stone, _reasoning), level) in enumerate(zip(entries, levels)):
        if method == "stone":
            requirement = stone or "Stone"
        elif method == "trade":
            requirement = "Trade"
        elif method == "friendship":
            requirement = "Friendship"
        else:
            requirement = f"Lv {level}"
        lines.append(f"Stage {i + 1}: {name} — {requirement}")
    return "\n".join(lines)


def build_evolution_line(gemini_client, character_name, wiki_context, candidate_titles):
    if not candidate_titles:
        return f"{character_name} has no evolution line."

    selection = select_real_stages(gemini_client, character_name, wiki_context, candidate_titles)
    if selection is None:
        return f"{character_name} has no evolution line."

    prevolutions, evolutions = [], []
    if selection.target_position == "stage1":
        evolutions = _verify_stages(selection.evolutions, candidate_titles)
    elif selection.target_position == "final":
        prevolutions = _verify_stages(selection.prevolutions, candidate_titles)
    else:
        prevolutions = _verify_stages(selection.prevolutions, candidate_titles)[:1]
        evolutions = _verify_stages(selection.evolutions, candidate_titles)[:1]

    if not prevolutions and not evolutions:
        return f"{character_name} has no evolution line."

    target_method, target_stone, target_reasoning = _verify_method_fields(
        selection.target_method, selection.target_stone, selection.target_reasoning
    )
    entries = _stage_entries(prevolutions, character_name, target_method, target_stone, target_reasoning, evolutions)

    known_tier = rarity_mode.check_known_game_rarity(character_name)
    return _format_line(entries, known_tier)
