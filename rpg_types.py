#!/usr/bin/env python3
"""Config for RPG Mode's battle system: Brainrot Types, Battle Types, and their match-up chart.

Edit these directly to rebalance — rpg_mode.py reads everything from here, so nothing else
needs to change when you do. Confirmed with the user before this was built (2026-08-26); Culinary
was added to BATTLE_TYPES shortly after based on direct feedback that a real Dex Entry example
(a food-themed character) had nothing to fit and defaulted to generic Physical for most of its
moves.
"""

# One per character — their habitat/identity, assigned from the character's lore/visual nature.
BRAINROT_TYPES = ["Aerial", "Land", "Sea", "Cosmic"]

# Used on moves and for battle match-up calculations (TYPE_CHART below). A character's Brainrot
# Type is a separate, smaller vocabulary (BRAINROT_TYPES above) — only two names overlap
# (Aerial, Cosmic) by coincidence of theme, not by design.
BATTLE_TYPES = [
    "Aquatic",  # water
    "Explosive",  # bombs, fire, detonation
    "Fortified",  # armor, structure, defensive bulk
    "Physical",  # brute force, muscle
    "Culinary",  # food, cooking, spice — added after the original 8 left food-themed
    # characters (a large share of this wiki) with no natural Battle Type; their moves were
    # defaulting to generic Physical instead of anything true to their actual theme.
    "Aerial",  # flight, wind
    "Musical",  # sound, rhythm
    "Cosmic",  # space, otherworldly
    "Attrition",  # wears opponents down over time
]

# Each Battle Type's strengths/weaknesses for damage calculations. Lists (not single strings) so
# a type can be given a 2nd strength/weakness later just by appending, without restructuring.
# A clean 9-way cycle — every type has exactly one strength and one weakness, so nothing is ever
# dominant or helpless:
#   Aquatic -> Explosive -> Fortified -> Physical -> Culinary -> Aerial -> Musical -> Cosmic ->
#   Attrition -> (back to Aquatic)
# Read "A -> B" as "A is strong against B". Confirmed with the user before finalizing.
TYPE_CHART = {
    "Aquatic": {"strong_against": ["Explosive"], "weak_against": ["Attrition"]},
    "Explosive": {"strong_against": ["Fortified"], "weak_against": ["Aquatic"]},
    "Fortified": {"strong_against": ["Physical"], "weak_against": ["Explosive"]},
    "Physical": {"strong_against": ["Culinary"], "weak_against": ["Fortified"]},
    "Culinary": {"strong_against": ["Aerial"], "weak_against": ["Physical"]},
    "Aerial": {"strong_against": ["Musical"], "weak_against": ["Culinary"]},
    "Musical": {"strong_against": ["Cosmic"], "weak_against": ["Aerial"]},
    "Cosmic": {"strong_against": ["Attrition"], "weak_against": ["Musical"]},
    "Attrition": {"strong_against": ["Aquatic"], "weak_against": ["Cosmic"]},
}

# A Brainrot Type's small passive: resists (takes reduced damage from) one Battle Type. Only
# Aerial and Cosmic map onto an identically-named Battle Type — Sea -> Aquatic and Land ->
# Physical are the closest thematic equivalents for the other two.
BRAINROT_TYPE_PASSIVES = {
    "Aerial": "Aerial",
    "Land": "Physical",
    "Sea": "Aquatic",
    "Cosmic": "Cosmic",
}
