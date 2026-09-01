#!/usr/bin/env python3
"""Config for RPG Mode's battle system: the 18 standard Pokémon-style types and their type
effectiveness chart.

Edit these directly to rebalance — rpg_mode.py reads everything from here, so nothing else
needs to change when you do.

History: originally a custom two-layer system (a small "Brainrot Type" habitat/identity layer
of 4 values, plus a separate 8-9 value "Battle Type" combat layer with its own cyclic strength/
weakness chart). Replaced with the standard 18 Pokémon types end to end — one shared vocabulary
for both a character's own type(s) and its moves' types, using the real, well-known Pokémon type
effectiveness chart instead of a custom cycle. The old habitat layer was dropped entirely rather
than kept as flavor (confirmed with the user): a character's 1-2 Pokémon Types already carry all
the identity/theme information Brainrot Type used to add, and real Pokémon itself has no
battle-facing "habitat" layer alongside type either.
"""

# The 18 standard types, Gen 6+ (current standard, post-Fairy). Used for both a character's own
# 1-2 types and every move's type — one shared vocabulary, matching real Pokémon.
POKEMON_TYPES = [
    "Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground",
    "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

# Sparse type effectiveness chart: TYPE_CHART[attacking_type][defending_type] = multiplier.
# Only non-1.0 entries are listed — any (attacker, defender) pair not present here is normal
# effectiveness (1.0), via effectiveness_multiplier() below. This is the standard, real Pokémon
# Gen 6+ chart verbatim (immune 0x / not very effective 0.5x / super effective 2x) — not a custom
# design, so rebalancing this means correcting a mistake against the real chart, not inventing
# new relationships. A sparse "only the exceptions" table is far easier to hand-edit and
# proofread than a dense 18x18 = 324-cell matrix would be.
TYPE_CHART = {
    "Normal": {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water": {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
    "Grass": {
        "Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5,
        "Rock": 2, "Dragon": 0.5, "Steel": 0.5,
    },
    "Ice": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
    "Fighting": {
        "Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Rock": 2,
        "Ghost": 0, "Dark": 2, "Steel": 2, "Fairy": 0.5,
    },
    "Poison": {"Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0, "Fairy": 2},
    "Ground": {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
    "Flying": {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
    "Bug": {
        "Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2, "Ghost": 0.5,
        "Dark": 2, "Steel": 0.5, "Fairy": 0.5,
    },
    "Rock": {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
    "Dragon": {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
    "Dark": {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Fairy": 0.5},
    "Steel": {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5, "Fairy": 2},
    "Fairy": {"Fire": 0.5, "Fighting": 2, "Poison": 0.5, "Dragon": 2, "Dark": 2, "Steel": 0.5},
}


def effectiveness_multiplier(attacking_type, defending_type):
    """The multiplier for one attacking type against one defending type — 1.0 (normal) for any
    pair not listed as an exception in TYPE_CHART above."""
    return TYPE_CHART.get(attacking_type, {}).get(defending_type, 1.0)


def combined_effectiveness(attacking_type, defending_types):
    """A move's total multiplier against a (possibly dual-typed) defender: the product of its
    effectiveness against each of the defender's types, same as real Pokémon (e.g. a Grass move
    against a Water/Ground defender is 2 x 2 = 4x)."""
    multiplier = 1.0
    for defending_type in defending_types:
        multiplier *= effectiveness_multiplier(attacking_type, defending_type)
    return multiplier
