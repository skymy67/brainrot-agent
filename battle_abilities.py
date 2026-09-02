#!/usr/bin/env python3
"""Config for RPG Mode's Battle Ability: the fixed set of mechanical, stat-based passive effect
categories a character's Battle Ability can be generated from, plus their tunable value ranges.

Edit these directly to rebalance — rpg_mode.py reads everything from here, so nothing else needs
to change when you do. Same role for Battle Ability that rpg_types.py plays for Pokémon Types:
one small, hand-editable source of truth the generator and its own validation code both read
from, so a hallucinated category name or an out-of-range value never reaches the output.

Battle Ability is deliberately kept separate from Signature Move (defined directly in
rpg_mode.py, no config file of its own): Battle Ability is a small, mechanical, closed-vocabulary
passive meant to plug straight into a future damage/turn calculation, while Signature Move is a
one-off, lore-derived active ability with no fixed category — closed-vocabulary config doesn't
fit it the same way.
"""

# The fixed set of Battle Ability effect categories. Each is a single, specific mechanical
# effect — never a combination of several — chosen by the model per character, then validated in
# rpg_mode.py against this exact list (a hallucinated/unmatched category name falls back to
# FALLBACK_CATEGORY below).
BATTLE_ABILITY_CATEGORIES = [
    "stat_boost",
    "damage_modifier",
    "status_effect",
    "passive_resistance",
    "healing",
    "priority",
]

# One-line prompt description per category, shown to the model when generating a Battle Ability.
# Keep each to a single clear mechanical effect — this is what keeps a generated ability usable
# directly in a future damage/turn calculation instead of reading as vague flavor text.
CATEGORY_DESCRIPTIONS = {
    "stat_boost": "raises one of the character's own stats under a trigger condition",
    "damage_modifier": (
        "increases or decreases the damage this character's own moves of a specific Pokémon "
        "Type deal"
    ),
    "status_effect": (
        "a percent chance to inflict a status on an opponent, or to resist one being inflicted "
        "on this character"
    ),
    "passive_resistance": "reduces the damage this character takes from a specific Pokémon Type",
    "healing": (
        "recovers a percentage of this character's max HP under a trigger condition "
        "(regen or lifesteal-style)"
    ),
    "priority": "lets this character act first (or last) in a turn under a trigger condition",
}

# (min, max) percent magnitude allowed per category — the range a generated ability's `magnitude`
# field is clamped into. "priority" has no numeric magnitude (it's a first/last-turn flag carried
# entirely by trigger_condition/effect_description), so its range is fixed at 0.
MAGNITUDE_RANGES = {
    "stat_boost": (10, 50),
    "damage_modifier": (10, 40),
    "status_effect": (10, 40),
    "passive_resistance": (10, 50),
    "healing": (5, 25),
    "priority": (0, 0),
}

# Safe fallback category for a hallucinated/unmatched category name — the simplest, lowest-risk
# effect to default to (a flat resistance needs no direction field, unlike damage_modifier or
# status_effect).
FALLBACK_CATEGORY = "passive_resistance"

# Fixed vocabulary for the status_effect category — Pokémon-style statuses reframed generically
# (no game-specific branding), since this is a passive on an original Battle Ability, not a move
# tied to one specific existing game's status system.
STATUS_EFFECTS = ["Paralysis", "Confusion", "Poison", "Sleep", "Flinch"]

# Which categories need a `direction` qualifier, and the fixed vocabulary for it. A category not
# listed here doesn't use direction at all (validated to "" in rpg_mode.py).
DIRECTION_OPTIONS_BY_CATEGORY = {
    "damage_modifier": ["increase", "decrease"],
    "status_effect": ["inflict", "resist"],
}
