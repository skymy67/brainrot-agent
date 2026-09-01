#!/usr/bin/env python3
"""RPG Mode: a compact Pokédex-style "Dex Entry" for a real wiki character.

Two independent pieces per request:
  1. A single structured Gemini call generates the character's 1-2 Pokémon Types, six battle
     stats (HP/Attack/Defense/Special Attack/Special Defense/Speed) — each with its own
     one-sentence reasoning, not just a bare number — 2-4 flavored moves (each with its own
     Pokémon Type, power, accuracy, and effect), and flavor text — grounded in wiki context,
     falling back to inferring from appearance/size/lore (and saying so, per-stat) when the wiki
     doesn't document a stat directly.
  2. A rarity tier, reusing rarity_mode.py's own scoring pipeline directly (its hardcoded-OG and
     known-game-rarity lookups first, cost-free; a search-grounded call only when neither hits) —
     not duplicated here, and not cached anywhere (this app has no persistence for that), so a
     character without a hardcoded/known rarity costs 2 extra Gemini calls beyond the Dex Entry
     call itself for its tier.

A character's own type(s) and every move's type are drawn from ONE shared fixed vocabulary — the
18 standard Pokémon types (rpg_types.POKEMON_TYPES) — given to the model in the prompt and
verified against that same list afterward — same defense-in-depth pattern evolution_mode.py uses
for its model-picked titles — so a hallucinated type name can never leak into the output.

History: this used to be a two-layer custom system (a small "Brainrot Type" habitat/identity
layer plus a separate "Battle Type" combat layer with its own custom effectiveness chart).
Replaced with the real, standard Pokémon type system end to end, per request — one vocabulary,
the well-known real effectiveness chart instead of a custom one, and the habitat layer dropped
entirely (confirmed with the user) rather than kept as flavor, since 1-2 real Pokémon Types
already carry all the identity information it used to add.
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
MAX_TYPES = 2
STAT_MIN, STAT_MAX = 1, 100
POWER_MIN, POWER_MAX = 1, 100
ACCURACY_MIN, ACCURACY_MAX = 1, 100
# Safe fallback type for a character or move whose model-picked type doesn't match the real
# list — Normal is the closest thing Pokémon has to a neutral default (no immunities, no unusual
# resistances either direction against it as an attacker).
FALLBACK_TYPE = "Normal"


class MoveOutput(BaseModel):
    name: str
    move_type: str
    power: int
    accuracy: int
    effect: str


class StatValue(BaseModel):
    value: int
    # Required, not optional — every stat must argue for itself, tying the number to a specific
    # documented detail (an ability, a combat feat, a described size) or, when the wiki doesn't
    # document it, explicitly saying it was inferred from appearance/size and why.
    reasoning: str


class StatsOutput(BaseModel):
    hp: StatValue
    attack: StatValue
    defense: StatValue
    special_attack: StatValue
    special_defense: StatValue
    speed: StatValue


class DexEntryOutput(BaseModel):
    types: list[str]  # 1-2 entries after validation
    flavor_text: str
    stats: StatsOutput
    moves: list[MoveOutput]
    # "" when the type(s) and moves were all grounded in documented wiki lore; otherwise a short
    # note on what else had to be inferred and why — never invented silently. Per-stat inference
    # reasoning lives on each StatValue instead; this covers everything else.
    inferred_note: str = ""


SYSTEM_INSTRUCTION = (
    "You are an RPG stat and battle-move generator for the Italian Brainrot universe, producing "
    "data for a compact Pokédex-style entry. Given wiki context about a character, generate:\n"
    "1) 1 to 2 Pokémon Types — from the fixed list given — based on the character's lore and "
    "appearance (e.g. a shark-based character could be Water type; a space-themed character "
    "could be Psychic or Dragon). Use a single type unless the character genuinely has two "
    "distinct thematic elements (like a real dual-type Pokémon) — don't force a second type just "
    "to fill the slot.\n"
    "2) six battle stats (HP, Attack, Defense, Special Attack, Special Defense, Speed), each on "
    "a 1-100 scale, based on documented abilities, combat feats, size, and power scaling from "
    "the context. EVERY stat needs its own one-sentence reasoning tying the number to a specific "
    "detail from the context (an ability, a combat feat, a described size or trait) — never a "
    "bare number with no justification. When the wiki doesn't document a stat directly, infer it "
    "from physical appearance, size, and apparent strength instead, and say so explicitly in "
    "that stat's own reasoning rather than inventing unstated facts silently. Keep stats "
    "consistent with comparable characters; don't inflate everything to the max.\n"
    "3) 2 to 4 signature moves, each with a short name, a Pokémon Type (from the fixed list "
    "given — moves don't have to share the character's own type(s), just like real Pokémon "
    "movesets), a power value (1-100), an accuracy percentage (typically 70-100, lower for "
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
        f"Pokémon Types (pick 1-2 for the character; moves may use any of these too): "
        f"{', '.join(rpg_types.POKEMON_TYPES)}\n\n"
        f"Generate the Dex Entry data per your instructions."
    )


def _match_from_list(value, valid_values, default):
    """Case-insensitive match against a fixed vocabulary, falling back to a safe default rather
    than trusting a hallucinated type name straight through."""
    lookup = {v.lower(): v for v in valid_values}
    return lookup.get((value or "").strip().lower(), default)


def _match_types(values, valid_values, default):
    """Same case-insensitive matching as _match_from_list, but for the character's own 1-2
    types: drops anything that doesn't match the real list, de-duplicates while preserving
    order, caps at MAX_TYPES, and falls back to [default] only if nothing valid is left."""
    lookup = {v.lower(): v for v in valid_values}
    matched = []
    for value in values:
        real = lookup.get((value or "").strip().lower())
        if real and real not in matched:
            matched.append(real)
    return matched[:MAX_TYPES] if matched else [default]


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

    character_types = _match_types(parsed.types, rpg_types.POKEMON_TYPES, FALLBACK_TYPE)

    def _clamp_stat(stat):
        return StatValue(value=_clamp(stat.value, STAT_MIN, STAT_MAX, 50), reasoning=(stat.reasoning or "").strip())

    stats = StatsOutput(
        hp=_clamp_stat(parsed.stats.hp),
        attack=_clamp_stat(parsed.stats.attack),
        defense=_clamp_stat(parsed.stats.defense),
        special_attack=_clamp_stat(parsed.stats.special_attack),
        special_defense=_clamp_stat(parsed.stats.special_defense),
        speed=_clamp_stat(parsed.stats.speed),
    )

    moves = [
        MoveOutput(
            name=(move.name or "Unnamed Move").strip(),
            move_type=_match_from_list(move.move_type, rpg_types.POKEMON_TYPES, FALLBACK_TYPE),
            power=_clamp(move.power, POWER_MIN, POWER_MAX, 50),
            accuracy=_clamp(move.accuracy, ACCURACY_MIN, ACCURACY_MAX, 90),
            effect=(move.effect or "").strip(),
        )
        for move in parsed.moves[:MAX_MOVES]
    ]
    # A model that returns fewer than MIN_MOVES despite instructions still gets shown as-is
    # rather than padded with invented filler moves — an honest short list beats a fabricated one.

    return DexEntryOutput(
        types=character_types,
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


# (label, StatsOutput attribute name) — spelled out in full rather than abbreviated (HP/ATK/DEF/
# SPA/SPDEF/SPD), per direct feedback that abbreviations made the stat block harder to scan.
STAT_LABELS = [
    ("HP", "hp"),
    ("Attack", "attack"),
    ("Defense", "defense"),
    ("Special Attack", "special_attack"),
    ("Special Defense", "special_defense"),
    ("Speed", "speed"),
]


def format_dex_entry(character_name, dex: DexEntryOutput, tier):
    stats = dex.stats
    move_lines = "\n".join(
        f"- {move.name} ({move.move_type}) | PWR {move.power} | ACC {move.accuracy}% — {move.effect}"
        for move in dex.moves
    )
    type_label = "/".join(dex.types)  # e.g. "Water" or "Water/Flying", standard dual-type notation
    lines = [f"**{character_name}** — {type_label} Type", ""]
    for label, attr in STAT_LABELS:
        stat = getattr(stats, attr)
        lines.append(f"**{label}:** {stat.value}/100 — {stat.reasoning}")
    lines.append("")
    lines.append(f"Rarity: {tier}")
    lines.append("")
    lines.append("Moves:")
    lines.append(move_lines if move_lines else "(none generated)")
    if dex.inferred_note:
        lines.append("")
        lines.append(f"*Inferred: {dex.inferred_note}*")
    # The flavor text is the Dex "entry" proper — placed last, bolded, per direct feedback
    # asking for it to close out the card instead of sitting right under the header.
    if dex.flavor_text:
        lines.append("")
        lines.append(f"**{dex.flavor_text}**")
    return "\n".join(lines)


def build_dex_entry(gemini_client, character_name, wiki_context):
    dex = generate_dex_data(gemini_client, character_name, wiki_context)
    if dex is None:
        return f"Couldn't generate a Dex Entry for {character_name} — try asking again."
    tier = compute_rarity_tier(gemini_client, character_name, wiki_context)
    return format_dex_entry(character_name, dex, tier)
