#!/usr/bin/env python3
"""Tunable config for Weakness Finder's documented in-wiki relationship extraction (friends,
enemies/rivals, family) and the canon-rivalry damage bonus. Edit these directly to adjust
extraction keywords, scan windows, or the bonus value without touching weakness_mode.py's logic.
"""

# Keyword groups matched as "<keyword> with|of <name>" (friend/enemy) or "<keyword> of <name>"
# (family — extends evolution_mode.py's own FAMILY_RELATION_RE list with uncle/aunt, which that
# module doesn't need but relationship extraction does).
FRIEND_KEYWORDS = ["friend", "friends", "ally", "allies", "allied"]
ENEMY_KEYWORDS = ["enemy", "enemies", "rival", "rivals", "nemesis", "nemeses"]
FAMILY_KEYWORDS = [
    "daughter", "daughters", "son", "sons", "sister", "sisters", "brother", "brothers",
    "wife", "wives", "husband", "husbands", "cousin", "cousins", "niece", "nieces",
    "nephew", "nephews", "mother", "mothers", "father", "fathers", "uncle", "uncles",
    "aunt", "aunts", "child", "children",
]

# How much of a character's own wiki content gets scanned for relationship mentions. Measured
# against the real corpus: relationship mentions ("best friends with X", "rivals with X") often
# sit well past the infobox/intro, and 99% of pages are under ~5200 characters total, so 8000
# comfortably covers nearly every real page in one pass — this runs once per request on a single
# character's content, so the extra scan cost is negligible.
RELATIONSHIP_CHECK_WINDOW = 8000
# How far past a relationship trigger phrase ("friends with", "uncle of", ...) to search for a
# real wiki title naming the other character — kept short so the match window doesn't run into an
# unrelated later sentence.
NAME_SEARCH_WINDOW = 120
# Caps how many distinct relationships of one type get shown, so one heavily-tagged page doesn't
# produce an overwhelming report.
MAX_PER_RELATIONSHIP_TYPE = 3

# Shown as an informational stat next to each documented enemy/rival. There's no live battle
# engine in this app to actually apply a damage bonus in a real fight, so this stays descriptive —
# grounded in a real documented canon rivalry, not a computed game mechanic.
RIVALRY_DAMAGE_BONUS_PERCENT = 25
