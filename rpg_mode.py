#!/usr/bin/env python3
"""RPG Mode: a compact Pokédex-style "Dex Entry" for a real wiki character.

Two independent pieces per request:
  1. A single structured Gemini call generates the character's 1-2 Pokémon Types, six battle
     stats (HP/Attack/Defense/Special Attack/Special Defense/Speed) — each with its own
     one-sentence reasoning, not just a bare number — 2-4 flavored moves (each with its own
     Pokémon Type, power, accuracy, and effect), a Battle Ability, a Signature Move, and flavor
     text — grounded in wiki context, falling back to inferring from appearance/size/lore (and
     saying so, per-stat) when the wiki doesn't document a stat directly.
  2. A rarity tier, reusing only rarity_mode.py's two free lookups (hardcoded-OG list, then
     known-game-rarity list) — no Gemini call, no cost. Deliberately does NOT fall through to
     Rarity Mode's own search-grounded guess for a character in neither list; that would add a
     second, heavier Gemini call (plus its own web-search tool call) on top of the Dex Entry
     call this mode already makes, tripling exposure to Gemini's transient-overload errors for a
     value that's a guess anyway. A character not in either free list shows "Unknown" instead —
     ask Rarity Mode directly if you want its full search-grounded estimate.

A character's own type(s) and every move's type are drawn from ONE shared fixed vocabulary — the
18 standard Pokémon types (rpg_types.POKEMON_TYPES) — given to the model in the prompt and
verified against that same list afterward — same defense-in-depth pattern evolution_mode.py uses
for its model-picked titles — so a hallucinated type name can never leak into the output.

Battle Ability vs. Signature Move: two deliberately different kinds of ability, generated
together in the same call but validated differently.
  - Battle Ability is mechanical and closed-vocabulary — generated from exactly one of
    battle_abilities.BATTLE_ABILITY_CATEGORIES (stat boost, damage modifier, status effect,
    passive resistance, healing, priority), each with its own tunable magnitude range defined in
    that config file. Every field (category, target stat/type/status, direction) is re-matched
    against its own fixed vocabulary in code — same defense-in-depth pattern as Pokémon Types —
    so it stays "usable directly in battle calculations" (structured, numeric, closed-vocabulary)
    even though no actual battle engine consumes it yet; this is presentational for now, shaped
    so a future one could.
  - Signature Move is the opposite on purpose: a one-off active ability with NO fixed category,
    generated from whatever standout trait (if any) is actually in this specific character's own
    wiki lore/personality/role. When no standout trait exists, the model is instructed to say so
    explicitly (is_generic + basis) rather than fabricate an elaborate ability to compensate.

History: this used to be a two-layer custom system (a small "Brainrot Type" habitat/identity
layer plus a separate "Battle Type" combat layer with its own custom effectiveness chart).
Replaced with the real, standard Pokémon type system end to end, per request — one vocabulary,
the well-known real effectiveness chart instead of a custom one, and the habitat layer dropped
entirely (confirmed with the user) rather than kept as flavor, since 1-2 real Pokémon Types
already carry all the identity information it used to add.
"""

from google.genai import types
from pydantic import BaseModel

import battle_abilities
import rarity_mode
import rpg_types
from content_policy import with_content_policy
from gemini_retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
DEX_THINKING_BUDGET = 640
# Bumped from 1536 alongside adding battle_ability + signature_move to the schema — more
# structured fields to fill in per response, same thinking budget.
DEX_MAX_OUTPUT_TOKENS = 2048
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


class BattleAbilityOutput(BaseModel):
    """A mechanical, stat-based passive effect, generated from exactly one of
    battle_abilities.BATTLE_ABILITY_CATEGORIES. Which target_* field is meaningful depends on the
    category (e.g. only stat_boost uses target_stat) — see _validate_battle_ability, which blanks
    out whichever fields don't apply to the chosen category rather than trusting the model to
    leave them empty on its own."""

    name: str
    category: str  # one of battle_abilities.BATTLE_ABILITY_CATEGORIES
    trigger_condition: str  # e.g. "always", "when HP is below 30%", "when hit by a Water-type move"
    target_stat: str = ""  # stat_boost only — one of the six stat keys (see STAT_LABELS)
    target_type: str = ""  # damage_modifier / passive_resistance only — a Pokémon Type
    target_status: str = ""  # status_effect only — one of battle_abilities.STATUS_EFFECTS
    # increase/decrease for damage_modifier, inflict/resist for status_effect; "" for every
    # other category (see battle_abilities.DIRECTION_OPTIONS_BY_CATEGORY).
    direction: str = ""
    magnitude: float = 0.0  # percent value, clamped to the category's MAGNITUDE_RANGES
    # One-sentence plain-language mechanical summary with the actual numbers in it, e.g.
    # "+20% Speed when HP is below 30%" — this is what gets displayed; the fields above exist so
    # the same ability is also usable programmatically by a future damage/turn calculation.
    effect_description: str


class SignatureMoveOutput(BaseModel):
    """A unique, named active ability tied to a standout trait from the character's own lore —
    deliberately NOT drawn from a fixed category list the way Battle Ability is, since the whole
    point is that it's specific to this one character."""

    name: str
    effect_description: str  # one concrete mechanical effect, e.g. bonus damage or a one-time buff
    power: int = 0  # 0 when the move isn't framed as a direct attack (e.g. a pure buff)
    accuracy: int = 0  # 0 when power is also 0
    # True when the wiki lore had no standout trait to draw from, so a simpler generic move was
    # generated instead of fabricating an elaborate one — set honestly, never left False by default
    # when the model itself had nothing distinctive to draw from.
    is_generic: bool = False
    # Required: the specific lore detail/catchphrase/role this move is drawn from, OR — when
    # is_generic is true — a short explanation of why no standout trait was available.
    basis: str = ""


class DexEntryOutput(BaseModel):
    types: list[str]  # 1-2 entries after validation
    flavor_text: str
    stats: StatsOutput
    moves: list[MoveOutput]
    battle_ability: BattleAbilityOutput
    signature_move: SignatureMoveOutput
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
    "4) flavor_text: exactly 1-2 sentences, in an in-universe Pokédex-entry tone.\n"
    "5) battle_ability: ONE mechanical passive ability that plugs directly into battle "
    "calculations, chosen from the fixed effect categories given in the prompt — pick exactly "
    "one category, never combine two. Fill in: a short thematic name; trigger_condition (e.g. "
    "'always', 'when HP is below 30%', 'when hit by a Water-type move'); whichever target field "
    "that category actually needs (a stat, a Pokémon Type, or a status, from the vocabularies "
    "given — leave the others blank); a direction where the category needs one (increase/"
    "decrease for damage_modifier, inflict/resist for status_effect); a magnitude value within "
    "that category's given range; and effect_description stating the exact numeric effect in "
    "plain language (e.g. '+20% Speed when HP is below 30%', '-25% damage taken from Water-type "
    "moves'). Thematically match the category and target to the character (e.g. a Water-type "
    "character resisting Water damage, a scrappy small character with a Speed boost when low on "
    "HP) — never pick a category or target arbitrarily.\n"
    "6) signature_move: ONE unique, named active ability tied specifically to a standout trait "
    "from THIS character's own wiki lore, personality, catchphrase, or role — something a "
    "generic template couldn't produce for a different character (e.g. a character known for "
    "commanding allies gets a reinforcements-style bonus-damage move; a character with a "
    "specific signature action in their lore gets that action turned into a mechanical effect). "
    "Give it one concrete mechanical effect (bonus damage, a one-time buff, or a special attack "
    "with its own power/accuracy) and cite the specific lore detail it's drawn from in `basis`. "
    "If the wiki context genuinely has no standout, distinctive trait to draw from, do NOT "
    "fabricate an elaborate move to compensate — set is_generic to true, generate a simpler, "
    "more generic active move instead, and say so explicitly in `basis` (e.g. 'No standout lore "
    "trait found for this character; a generic all-purpose strike was used instead')."
)
SYSTEM_INSTRUCTION = with_content_policy(SYSTEM_INSTRUCTION)


def _format_battle_ability_categories():
    lines = []
    for category in battle_abilities.BATTLE_ABILITY_CATEGORIES:
        low, high = battle_abilities.MAGNITUDE_RANGES[category]
        magnitude_note = "no numeric magnitude" if (low, high) == (0, 0) else f"magnitude range {low}-{high}%"
        directions = battle_abilities.DIRECTION_OPTIONS_BY_CATEGORY.get(category)
        direction_note = f", direction: {'/'.join(directions)}" if directions else ""
        lines.append(f"- {category}: {battle_abilities.CATEGORY_DESCRIPTIONS[category]} ({magnitude_note}{direction_note})")
    return "\n".join(lines)


def _build_prompt(character_name, wiki_context):
    stat_keys = ", ".join(attr for _, attr in STAT_LABELS)
    return (
        f"Context from the Italian Brainrot wiki:\n\n"
        f"{wiki_context or '(no wiki data found for this character)'}\n\n"
        f"Character: {character_name}\n\n"
        f"Pokémon Types (pick 1-2 for the character; moves may use any of these too): "
        f"{', '.join(rpg_types.POKEMON_TYPES)}\n\n"
        f"Battle Ability effect categories (pick exactly one):\n{_format_battle_ability_categories()}\n\n"
        f"Stat names, for the stat_boost category's target_stat only: {stat_keys}\n\n"
        f"Statuses, for the status_effect category's target_status only: "
        f"{', '.join(battle_abilities.STATUS_EFFECTS)}\n\n"
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


def _clamp_magnitude(value, category):
    low, high = battle_abilities.MAGNITUDE_RANGES.get(category, (0, 0))
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return float(low)


def _validate_battle_ability(raw: BattleAbilityOutput) -> BattleAbilityOutput:
    """Same defense-in-depth pattern as _match_types/_clamp above: every field is re-matched
    against its own fixed vocabulary (from battle_abilities.py) or numeric range, and any field
    that doesn't apply to the chosen category is forced blank rather than trusting the model to
    have left it that way on its own."""
    category = _match_from_list(raw.category, battle_abilities.BATTLE_ABILITY_CATEGORIES, battle_abilities.FALLBACK_CATEGORY)

    stat_keys = [attr for _, attr in STAT_LABELS]
    target_stat = _match_from_list(raw.target_stat, stat_keys, stat_keys[0]) if category == "stat_boost" else ""
    target_type = (
        _match_from_list(raw.target_type, rpg_types.POKEMON_TYPES, FALLBACK_TYPE)
        if category in ("damage_modifier", "passive_resistance")
        else ""
    )
    target_status = (
        _match_from_list(raw.target_status, battle_abilities.STATUS_EFFECTS, battle_abilities.STATUS_EFFECTS[0])
        if category == "status_effect"
        else ""
    )

    valid_directions = battle_abilities.DIRECTION_OPTIONS_BY_CATEGORY.get(category)
    direction = _match_from_list(raw.direction, valid_directions, valid_directions[0]) if valid_directions else ""

    return BattleAbilityOutput(
        name=(raw.name or "Unnamed Ability").strip(),
        category=category,
        trigger_condition=(raw.trigger_condition or "").strip() or "always",
        target_stat=target_stat,
        target_type=target_type,
        target_status=target_status,
        direction=direction,
        magnitude=_clamp_magnitude(raw.magnitude, category),
        effect_description=(raw.effect_description or "").strip(),
    )


def _validate_signature_move(raw: SignatureMoveOutput) -> SignatureMoveOutput:
    return SignatureMoveOutput(
        name=(raw.name or "Unnamed Move").strip(),
        effect_description=(raw.effect_description or "").strip(),
        power=_clamp(raw.power, 0, POWER_MAX, 0),
        accuracy=_clamp(raw.accuracy, 0, ACCURACY_MAX, 0),
        is_generic=bool(raw.is_generic),
        basis=(raw.basis or "").strip(),
    )


def generate_dex_data(gemini_client, character_name, wiki_context):
    """Runs the single structured Gemini call and returns a DexEntryOutput with every type name
    and numeric value already verified/clamped against the real vocabulary and valid ranges —
    callers never need to re-validate the result themselves. Returns None if the model's
    response didn't parse against the schema at all (e.g. cut off before finishing) — genuinely
    rare with schema-enforced output, but not impossible."""
    response = call_with_retry(lambda: gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(character_name, wiki_context),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=DEX_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=DEX_THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=DexEntryOutput,
        ),
    ))
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
        battle_ability=_validate_battle_ability(parsed.battle_ability),
        signature_move=_validate_signature_move(parsed.signature_move),
        inferred_note=(parsed.inferred_note or "").strip(),
    )


def compute_rarity_tier(character_name):
    """Reuses rarity_mode.py's two free lookups (hardcoded OG list, then known-game-rarity list)
    for the Dex Entry's compact stat block. Deliberately does NOT fall through to Rarity Mode's
    search-grounded guess for everything else — that's a live Gemini call (with its own
    web-search tool call layered on top), and RPG Mode already makes its own separate Gemini call
    for the rest of the Dex Entry; stacking a second, heavier call onto every character not in
    one of the two free lists tripled this mode's exposure to Gemini's own transient-overload
    errors (see gemini_retry.py) for a value that's a guess anyway. Returns None when neither
    free lookup hits — format_dex_entry shows "Unknown" rather than spending an extra call to
    guess at one."""
    if rarity_mode.check_hardcoded_og(character_name):
        return "OG"
    return rarity_mode.check_known_game_rarity(character_name)


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
    lines.append(f"Rarity: {tier or 'Unknown'}")
    lines.append("")
    lines.append(f"**Battle Ability:** {dex.battle_ability.name} — {dex.battle_ability.effect_description}")
    signature_note = " *(generic — " + dex.signature_move.basis + ")*" if dex.signature_move.is_generic and dex.signature_move.basis else ""
    lines.append(f"**Signature Move:** {dex.signature_move.name} — {dex.signature_move.effect_description}{signature_note}")
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
    tier = compute_rarity_tier(character_name)
    return format_dex_entry(character_name, dex, tier)
