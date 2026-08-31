#!/usr/bin/env python3
"""RPG Mode: a compact Pokédex-style "Dex Entry" for a real wiki character.

Two independent pieces per request:
  1. A single structured Gemini call generates the Brainrot Type, six battle stats (HP/Attack/
     Defense/Special Attack/Special Defense/Speed), 2-4 flavored moves (each with a Battle Type,
     power, accuracy, and effect), and flavor text — grounded in wiki context, falling back to
     inferring from appearance/size/lore (and saying so) when the wiki doesn't document a stat
     directly, the same reasoning style the original stat-only version of this mode already used.
  2. A rarity tier, reusing rarity_mode.py's own scoring pipeline directly (its hardcoded-OG and
     known-game-rarity lookups first, cost-free; a search-grounded call only when neither hits) —
     not duplicated here, and not cached anywhere (this app has no persistence for that), so a
     character without a hardcoded/known rarity costs 2 extra Gemini calls beyond the Dex Entry
     call itself for its tier.

Both a character's Brainrot Type and every move's Battle Type are drawn from a fixed vocabulary
given to the model in the prompt (rpg_types.BRAINROT_TYPES / BATTLE_TYPES) and verified against
that same list afterward — same defense-in-depth pattern evolution_mode.py uses for its
model-picked titles — so a hallucinated type name can never leak into the output.
"""

from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

import rarity_mode
import rpg_types

GEMINI_MODEL = "gemini-3.6-flash"
DEX_THINKING_BUDGET = 640
DEX_MAX_OUTPUT_TOKENS = 1536
MIN_MOVES = 2
MAX_MOVES = 4
STAT_MIN, STAT_MAX = 1, 100
POWER_MIN, POWER_MAX = 1, 100
ACCURACY_MIN, ACCURACY_MAX = 1, 100
# A safe fallback Battle Type for a move whose model-picked type doesn't match the real list —
# generic enough to apply to any character rather than guessing something theme-specific.
FALLBACK_BATTLE_TYPE = "Physical"
FALLBACK_BRAINROT_TYPE = "Land"


class MoveOutput(BaseModel):
    name: str
    battle_type: str
    power: int
    accuracy: int
    effect: str


class StatsOutput(BaseModel):
    hp: int
    attack: int
    defense: int
    special_attack: int
    special_defense: int
    speed: int


class DexEntryOutput(BaseModel):
    brainrot_type: str
    flavor_text: str
    stats: StatsOutput
    moves: list[MoveOutput]
    # "" when every stat/type was grounded in documented wiki lore; otherwise a short note on
    # what had to be inferred from appearance/size and why — never invented silently.
    inferred_note: str = ""


SYSTEM_INSTRUCTION = (
    "You are an RPG stat and battle-move generator for the Italian Brainrot universe, producing "
    "data for a compact Pokédex-style entry. Given wiki context about a character, generate:\n"
    "1) a Brainrot Type — exactly one, from the fixed list given, based on the character's "
    "habitat/visual nature (where it lives or what element it visually belongs to).\n"
    "2) six battle stats (HP, Attack, Defense, Special Attack, Special Defense, Speed), each on "
    "a 1-100 scale, based on documented abilities, combat feats, size, and power scaling from "
    "the context. When the wiki doesn't document a stat directly, infer it from physical "
    "appearance, size, and apparent strength instead — and say so explicitly in inferred_note "
    "rather than inventing unstated facts silently. Keep stats consistent with comparable "
    "characters; don't inflate everything to the max.\n"
    "3) 2 to 4 signature moves, each with a short name, a Battle Type (from the fixed list "
    "given), a power value (1-100), an accuracy percentage (typically 70-100, lower for "
    "high-risk moves), and a one-line effect description. Moves must be flavored to this "
    "specific character's lore, personality, and appearance — never generic filler moves a "
    "different character could equally have.\n"
    "4) flavor_text: exactly 1-2 sentences, in an in-universe Pokédex-entry tone."
)


def _build_prompt(character_name, wiki_context):
    return (
        f"Context from the Italian Brainrot wiki:\n\n"
        f"{wiki_context or '(no wiki data found for this character)'}\n\n"
        f"Character: {character_name}\n\n"
        f"Brainrot Types (pick exactly one): {', '.join(rpg_types.BRAINROT_TYPES)}\n"
        f"Battle Types (use only these for moves): {', '.join(rpg_types.BATTLE_TYPES)}\n\n"
        f"Generate the Dex Entry data per your instructions."
    )


def _match_from_list(value, valid_values, default):
    """Case-insensitive match against a fixed vocabulary, falling back to a safe default rather
    than trusting a hallucinated type name straight through."""
    lookup = {v.lower(): v for v in valid_values}
    return lookup.get((value or "").strip().lower(), default)


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def generate_dex_data(gemini_client, character_name, wiki_context):
    """Runs the single structured Gemini call and returns a DexEntryOutput with every type name
    and numeric value already verified/clamped against the real vocabulary and valid ranges —
    callers never need to re-validate the result themselves. Returns None if the model's
    response didn't parse against the schema at all (e.g. cut off before finishing) — genuinely
    rare with schema-enforced output, but not impossible."""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(character_name, wiki_context),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=DEX_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=DEX_THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=DexEntryOutput,
        ),
    )
    parsed = response.parsed
    if parsed is None:
        return None

    brainrot_type = _match_from_list(parsed.brainrot_type, rpg_types.BRAINROT_TYPES, FALLBACK_BRAINROT_TYPE)

    stats = StatsOutput(
        hp=_clamp(parsed.stats.hp, STAT_MIN, STAT_MAX, 50),
        attack=_clamp(parsed.stats.attack, STAT_MIN, STAT_MAX, 50),
        defense=_clamp(parsed.stats.defense, STAT_MIN, STAT_MAX, 50),
        special_attack=_clamp(parsed.stats.special_attack, STAT_MIN, STAT_MAX, 50),
        special_defense=_clamp(parsed.stats.special_defense, STAT_MIN, STAT_MAX, 50),
        speed=_clamp(parsed.stats.speed, STAT_MIN, STAT_MAX, 50),
    )

    moves = [
        MoveOutput(
            name=(move.name or "Unnamed Move").strip(),
            battle_type=_match_from_list(move.battle_type, rpg_types.BATTLE_TYPES, FALLBACK_BATTLE_TYPE),
            power=_clamp(move.power, POWER_MIN, POWER_MAX, 50),
            accuracy=_clamp(move.accuracy, ACCURACY_MIN, ACCURACY_MAX, 90),
            effect=(move.effect or "").strip(),
        )
        for move in parsed.moves[:MAX_MOVES]
    ]
    # A model that returns fewer than MIN_MOVES despite instructions still gets shown as-is
    # rather than padded with invented filler moves — an honest short list beats a fabricated one.

    return DexEntryOutput(
        brainrot_type=brainrot_type,
        flavor_text=(parsed.flavor_text or "").strip(),
        stats=stats,
        moves=moves,
        inferred_note=(parsed.inferred_note or "").strip(),
    )


def compute_rarity_tier(gemini_client, character_name, wiki_context):
    """Reuses rarity_mode.py's own pipeline stages directly (hardcoded OG list, then known-
    game-rarity list — both free lookups — then its search-grounded fallback) to get just the
    tier string for the Dex Entry's compact stat block, rather than Rarity Mode's own full
    formatted explanation paragraph, which belongs to Rarity Mode's own output, not this one."""
    if rarity_mode.check_hardcoded_og(character_name):
        return "OG"

    known = rarity_mode.check_known_game_rarity(character_name)
    if known:
        return known

    try:
        search_result = rarity_mode.run_og_and_rarity_search(gemini_client, character_name, wiki_context)
    except genai_errors.APIError as exc:
        # Mirrors rarity_mode.run_rarity_pipeline's own fallback for this exact failure —
        # constructed directly rather than reaching into its private _unavailable_search_result
        # helper, using only the module's public class/function surface.
        reason = f"API error {exc.code}"
        search_result = rarity_mode.OgAndRaritySearchResult(
            is_og=False,
            og_reasoning=f"Could not determine OG status ({reason}).",
            existing_game_rarity_found=False,
            existing_game_rarity="",
            existing_game_rarity_reasoning=f"Existing-rarity search unavailable ({reason}).",
        )

    if search_result.is_og:
        return "OG"

    score_result = rarity_mode.run_final_scoring(gemini_client, character_name, wiki_context, search_result, None, None)
    return rarity_mode.score_to_tier(score_result.total_score, score_result.categories_maxed)


def format_dex_entry(character_name, dex: DexEntryOutput, tier):
    stats = dex.stats
    move_lines = "\n".join(
        f"- {move.name} ({move.battle_type}) | PWR {move.power} | ACC {move.accuracy}% — {move.effect}"
        for move in dex.moves
    )
    lines = [
        f"**{character_name}** — {dex.brainrot_type} Type",
    ]
    if dex.flavor_text:
        lines.append(dex.flavor_text)
    lines.append("")
    lines.append(
        f"HP {stats.hp} | ATK {stats.attack} | DEF {stats.defense} | "
        f"SPA {stats.special_attack} | SPDEF {stats.special_defense} | SPD {stats.speed}"
    )
    lines.append(f"Rarity: {tier}")
    lines.append("")
    lines.append("Moves:")
    lines.append(move_lines if move_lines else "(none generated)")
    if dex.inferred_note:
        lines.append("")
        lines.append(f"*Inferred: {dex.inferred_note}*")
    return "\n".join(lines)


def build_dex_entry(gemini_client, character_name, wiki_context):
    dex = generate_dex_data(gemini_client, character_name, wiki_context)
    if dex is None:
        return f"Couldn't generate a Dex Entry for {character_name} — try asking again."
    tier = compute_rarity_tier(gemini_client, character_name, wiki_context)
    return format_dex_entry(character_name, dex, tier)
