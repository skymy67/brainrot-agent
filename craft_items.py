#!/usr/bin/env python3
"""Config for Craft Mode's item system: which components become collectible items, what stat
effect each item category grants, and how an item's rarity tier is computed.

Edit these directly to rebalance — craft_mode.py reads everything from here, so nothing else
needs to change when you do. Same role this plays for Craft Mode's items that rpg_types.py plays
for Pokémon Types and battle_abilities.py plays for Battle Ability: a small, hand-editable source
of truth the generator's validation code reads from, so a hallucinated category, stat, or
out-of-range magnitude never reaches the output.

Only "item"-type components (objects and food) get stat effects and a rarity — an "animal"
component (the character's base creature) stays pure lore, no item stats, per the spec's own
distinction between "part of the character's identity" and "a collectible component."

Item rarity is deterministic and code-only — no extra Gemini call, same "no-extra-call"
philosophy as rpg_mode.compute_rarity_tier: it's computed from the item's own stat magnitude,
how many effects it has, and (for free, via rarity_mode.py's two free lookups only — never its
search-grounded guess) the source character's own rarity tier if already known.
"""

# --- Component classification -------------------------------------------------------------

# Top-level split: only "item" components (objects/food) get item_category/effects — "animal"
# components (the character's base creature) are pure lore, shown with no stats.
COMPONENT_TYPES = ["animal", "item"]

# The six stat-effect categories an "item" component gets classified into. Each maps to one
# primary stat (or, for "instrument", a choice between two) per the spec's own mapping.
ITEM_CATEGORIES = ["footwear", "weapon", "armor", "instrument", "reflective", "food"]
# Safe fallback category for a hallucinated/unmatched category name.
FALLBACK_CATEGORY = "armor"

# Not one of RPG Mode's six stats — a distinct effect kind (a percent chance to inflict a
# status), only ever paired with the "instrument" category as an alternative to Special Attack.
STATUS_CHANCE_STAT = "status_chance"

# RPG Mode's six battle stats, duplicated here (rather than imported from rpg_mode.py) so this
# config file stays fully self-contained and editable on its own — must stay in sync with
# rpg_mode.STAT_LABELS' own attribute keys.
STAT_KEYS = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]
STAT_DISPLAY_NAMES = {
    "hp": "HP",
    "attack": "Attack",
    "defense": "Defense",
    "special_attack": "Special Attack",
    "special_defense": "Special Defense",
    "speed": "Speed",
}

# Which stat(s) each item category's PRIMARY effect may use. A list because "instrument" is the
# one category where the model picks between two options (per the spec's "Special Attack boost
# OR status-effect chance"); every other category has exactly one valid stat.
CATEGORY_PRIMARY_STATS = {
    "footwear": ["speed"],
    "weapon": ["attack"],
    "armor": ["defense"],
    "instrument": ["special_attack", STATUS_CHANCE_STAT],
    "reflective": ["special_defense"],
    "food": ["hp"],
}

# Only "food" items get an optional secondary effect (per the spec's "small optional secondary
# Special Defense bonus") — always on this one fixed stat, never model-chosen.
SECONDARY_STAT_BY_CATEGORY = {"food": "special_defense"}

# --- Magnitude ranges ------------------------------------------------------------------------

# (min, max) stat bonus an item category's PRIMARY effect can be clamped into. Weapon/armor get
# a higher ceiling than the lighter accessory categories — purely a game-balance choice, easy to
# retune here.
MAGNITUDE_RANGES = {
    "footwear": (5, 20),
    "weapon": (10, 30),
    "armor": (10, 25),
    "instrument": (5, 20),
    "reflective": (5, 20),
    "food": (5, 20),
}
# The food-only secondary bonus is always small relative to the primary HP bonus.
SECONDARY_MAGNITUDE_RANGE = (2, 10)
# A status_chance effect's magnitude is a percent, not a stat point value.
STATUS_CHANCE_MAGNITUDE_RANGE = (5, 30)

# --- Rarity scoring --------------------------------------------------------------------------

# Converts a primary effect's magnitude into 0-4 points, normalized against that category's own
# MAGNITUDE_RANGES span (so a "big for footwear" item and a "big for weapon" item score the same
# despite having different raw ranges) — (max_normalized_fraction, points), checked in order,
# first match wins.
MAGNITUDE_POINT_BUCKETS = [
    (0.25, 0),
    (0.50, 1),
    (0.75, 2),
    (0.90, 3),
    (1.01, 4),
]

# A multi-effect item (currently only possible for "food", via its optional secondary) ranks
# higher than a single-effect one, per the spec.
EFFECT_COUNT_BONUS = {1: 0, 2: 2}

# A small boost from the source character's OWN rarity tier, looked up for free via
# rarity_mode.py's hardcoded-OG and known-game-rarity lookups only (never its search-grounded
# guess — matches rpg_mode.compute_rarity_tier's own no-extra-Gemini-call rule). A character
# whose tier isn't cheaply known contributes no bonus.
CHARACTER_TIER_BONUS = {
    "Common": 0, "Uncommon": 0, "Rare": 0, "Epic": 0,
    "Legendary": 1, "Mythic": 1,
    "Brainrot God": 2, "Secret": 2, "Celestial": 2,
    "OG": 2,
}

# total_score = magnitude_points (0-4) + EFFECT_COUNT_BONUS (0 or 2) + CHARACTER_TIER_BONUS
# (0-2), max 8 — mapped 1:1 onto the 9 non-OG Rarity Mode tiers. The top three additionally
# require a multi-effect item, not just a high score, same "reserve the top tiers for signal
# spread, not one maxed-out factor" idea rarity_mode.py's own SCORE_THRESHOLDS uses.
#
# (min_score, requires_multi_effect, tier), checked HIGHEST to LOWEST — the first entry whose
# min_score the item's score clears (and whose multi-effect requirement, if any, is satisfied)
# wins. This floor-based, highest-first shape is what lets a single-effect item that scores high
# enough for a multi-gated tier gracefully fall through to the best tier it CAN reach (capped at
# Mythic) instead of falling all the way through to Common — a fixed (min, max) window per tier
# left no bucket at all for "scores 6-8 without the multi-effect it'd need," which is a real,
# reachable case (verified directly: a single max-magnitude item from a high-tier character
# scores 6 on its own).
RARITY_SCORE_THRESHOLDS = [
    (8, True, "Celestial"),
    (7, True, "Secret"),
    (6, True, "Brainrot God"),
    (5, False, "Mythic"),
    (4, False, "Legendary"),
    (3, False, "Epic"),
    (2, False, "Rare"),
    (1, False, "Uncommon"),
    (0, False, "Common"),
]
# An item from a hardcoded-OG character skips scoring entirely and is always "OG" — mirrors
# rarity_mode.py's own OG-short-circuits-before-scoring structure.
