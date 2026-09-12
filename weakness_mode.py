#!/usr/bin/env python3
"""Weakness Finder: given a character, determines its Pokémon Type(s) and shows a full
weakness/resistance/immunity breakdown using the real Pokémon type-effectiveness chart in
rpg_types.py — data that's been sitting there unused by every mode until now.

Also suggests a real "rival": a genuine wiki character retrieved via a second, free local RAG
query keyed on the character's single most-effective countering type, never a model free-recall.
Evolution Mode's own docstring is explicit about why: letting a model freely recall a name from
its own training data invites hallucination, even with instructions not to. The rival here is
picked directly from real retrieval results (the top match, excluding the target itself) — a
mechanical, zero-hallucination-risk pick, not an LLM judgment call — so the whole mode costs
exactly ONE Gemini call (type determination only), matching every other single-shot mode's cost.

Type-effectiveness math itself is never left to the model either — rpg_types.combined_effectiveness
is deterministic, already-trusted code, so asking an LLM to reproduce that reasoning would just
introduce a chance of getting the real chart wrong for no benefit.

app.py orchestrates the two sequential retrievals directly (same pattern it already uses for
Evolution Mode's own multi-query candidate retrieval): it retrieves the target's own context,
calls determine_type(), computes the countering type via top_weak_type(), retrieves rival
candidates with rival_query() as the search string, then calls format_weakness_report() with
whatever pick_rival() found. This module stays retrieval-agnostic — it never touches chroma_db
itself.

Also surfaces documented in-wiki RELATIONSHIPS — friends, enemies/rivals, and family — found via
find_relationships(). This is pure code, pattern-matched directly against the character's own
wiki prose (relationship_config.py's keyword lists), costing no extra Gemini call and carrying no
hallucination risk: a relationship is only reported when the text right after a trigger phrase
("friends with", "uncle of", ...) names a REAL wiki page, verified against akinator_mode.ALL_TITLES
— the same "never invent, always verify against a real vocabulary" rule every other mode follows.
A documented enemy/rival additionally gets a stated canon-rivalry damage-bonus stat
(relationship_config.RIVALRY_DAMAGE_BONUS_PERCENT) — informational only, since this app has no
live battle engine to actually apply it in a real fight.
"""

import re

from google.genai import types
from pydantic import BaseModel

import akinator_mode
import relationship_config as rel_config
import rpg_mode
import rpg_types
from content_policy import with_content_policy
from gemini_retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 256
MAX_OUTPUT_TOKENS = 512


class TypeDetermination(BaseModel):
    types: list[str] = []  # 1-2 entries, from rpg_types.POKEMON_TYPES


SYSTEM_INSTRUCTION = (
    "You determine a character's 1-2 Pokémon Types for a type-weakness lookup, based on its "
    "lore and appearance — the same fixed type list and grounding rules RPG Mode uses (e.g. a "
    "shark-based character could be Water type; a space-themed character could be Psychic or "
    "Dragon). Use a single type unless the character genuinely has two distinct thematic "
    "elements — don't force a second type just to fill the slot."
)
SYSTEM_INSTRUCTION = with_content_policy(SYSTEM_INSTRUCTION)


def _build_prompt(character_name, wiki_context):
    return (
        f"Context from the Italian Brainrot wiki:\n\n"
        f"{wiki_context or '(no wiki data found for this character)'}\n\n"
        f"Character: {character_name}\n\n"
        f"Pokémon Types (pick 1-2): {', '.join(rpg_types.POKEMON_TYPES)}\n\n"
        f"Determine this character's type(s) per your instructions."
    )


def determine_type(gemini_client, character_name, wiki_context):
    """Runs the single structured Gemini call and returns a validated list of 1-2 real Pokémon
    Types (never a hallucinated one — same _match_types-style defense-in-depth rpg_mode.py uses),
    or None if the response didn't parse against the schema at all."""
    response = call_with_retry(lambda: gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_build_prompt(character_name, wiki_context),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=TypeDetermination,
        ),
    ))
    parsed = response.parsed
    if parsed is None:
        return None

    lookup = {t.lower(): t for t in rpg_types.POKEMON_TYPES}
    matched = []
    for raw_type in parsed.types:
        real = lookup.get((raw_type or "").strip().lower())
        if real and real not in matched:
            matched.append(real)
    return matched[:2] if matched else [rpg_mode.FALLBACK_TYPE]


def top_weak_type(defending_types):
    """The single attacking type with the highest combined_effectiveness against the given
    defending types — i.e. the type a rival would most want to be, to hit this character hardest.
    Returns None if nothing exceeds neutral (1x), meaning there's no real weakness to build a
    rival suggestion from. Ties broken by rpg_types.POKEMON_TYPES order, for determinism."""
    best_type, best_multiplier = None, 1.0
    for attacking_type in rpg_types.POKEMON_TYPES:
        multiplier = rpg_types.combined_effectiveness(attacking_type, defending_types)
        if multiplier > best_multiplier:
            best_type, best_multiplier = attacking_type, multiplier
    return best_type


def rival_query(weak_type):
    """The free local RAG query used to retrieve real rival candidates for a given countering
    type — no wiki page literally says 'Water type', so this leans on the embedding model's own
    semantic sense of the type name and common thematic words for it, same imprecision every
    other mode's own RAG-based candidate retrieval already accepts."""
    return f"{weak_type}-type themed brainrot character with {weak_type.lower()} powers or imagery"


def pick_rival(rival_metadatas, exclude_title):
    """Picks the top real retrieval result that isn't the target character itself — a mechanical
    choice, not an LLM judgment call, so there's no hallucination risk: the result is always a
    literal title that came back from a real chroma_db query, or None if nothing else qualified."""
    exclude_lower = exclude_title.strip().lower()
    for meta in rival_metadatas:
        if meta["title"].strip().lower() != exclude_lower:
            return meta["title"]
    return None


def _relationship_pattern(keywords, connectors):
    keyword_alt = "|".join(re.escape(k) for k in keywords)
    connector_alt = "|".join(connectors)
    return re.compile(rf"(?i)\b(?:{keyword_alt})\s+(?:{connector_alt})\b")


# family relations are always phrased "X of Y" in this wiki ("uncle of X", "children of X");
# friend/enemy relations use "with" in every real example found, "of" kept too for robustness.
FRIEND_RE = _relationship_pattern(rel_config.FRIEND_KEYWORDS, ["with", "of"])
ENEMY_RE = _relationship_pattern(rel_config.ENEMY_KEYWORDS, ["with", "of"])
FAMILY_RE = _relationship_pattern(rel_config.FAMILY_KEYWORDS, ["of"])
RELATIONSHIP_PATTERNS = [("friend", FRIEND_RE), ("enemy", ENEMY_RE), ("family", FAMILY_RE)]


def _find_title_in_window(window, exclude_title):
    """Verifies a relationship match names a REAL wiki page — never trusts free text alone. Scans
    akinator_mode.ALL_TITLES (already loaded once at startup) for any real title appearing as a
    whole word/phrase in the short text window right after a relationship trigger, preferring
    whichever real title starts EARLIEST in the window — i.e. the one immediately named by the
    trigger phrase — since the window can span into an unrelated later sentence that happens to
    also name a real (but irrelevant) character. Ties at the same start position prefer the
    longer title (so e.g. 'Tralalero Tralala' wins over a shorter title that's a substring of
    it)."""
    exclude_lower = exclude_title.strip().lower()
    window_lower = window.lower()
    best_title, best_key = None, None
    for title in akinator_mode.ALL_TITLES:
        title_lower = title.strip().lower()
        if title_lower == exclude_lower or title_lower not in window_lower:
            continue
        match = re.search(r"(?i)\b" + re.escape(title) + r"\b", window)
        if not match:
            continue
        key = (match.start(), -len(title))
        if best_key is None or key < best_key:
            best_key, best_title = key, title
    return best_title


def find_relationships(character_name, content):
    """Scans a character's own wiki content for documented friend/enemy/family relationships with
    other REAL wiki characters. Pattern-matched directly against real wiki prose (no Gemini call,
    so zero hallucination risk beyond the regex itself) — a relationship is only kept when the
    text right after the trigger phrase actually names a real wiki page via _find_title_in_window,
    never a guessed or partially-matched name. The search window is cut off at the first sentence
    boundary (. ! ?) — a real find must be in the SAME clause as the trigger; without this, a typo
    in the intended name (breaking the exact-title match) can let the lookup fall through to an
    unrelated real name later in the window, and falsely attribute it as the relationship (found
    on real wiki text: 'wife of Cappucino Assassino' — a misspelling of Cappuccino Assassino —
    otherwise let a later, unrelated 'Ballerino Lololo' mention get reported as her family).
    Returns {"friend": [...], "enemy": [...], "family": [...]}, each a list of real titles, capped
    per type by MAX_PER_RELATIONSHIP_TYPE."""
    window_text = (content or "")[:rel_config.RELATIONSHIP_CHECK_WINDOW]
    found = {"friend": [], "enemy": [], "family": []}
    seen = {"friend": set(), "enemy": set(), "family": set()}
    for relation_type, pattern in RELATIONSHIP_PATTERNS:
        for match in pattern.finditer(window_text):
            if len(found[relation_type]) >= rel_config.MAX_PER_RELATIONSHIP_TYPE:
                break
            search_window = window_text[match.end():match.end() + rel_config.NAME_SEARCH_WINDOW]
            sentence_end = re.search(r"[.!?]", search_window)
            if sentence_end:
                search_window = search_window[:sentence_end.start()]
            other_title = _find_title_in_window(search_window, character_name)
            if other_title and other_title not in seen[relation_type]:
                seen[relation_type].add(other_title)
                found[relation_type].append(other_title)
    return found


def format_relationships(relationships):
    """Renders find_relationships()'s output as report lines. The enemy/rival line carries the
    stated canon-rivalry damage bonus — see the module docstring for why that's informational
    rather than a computed game mechanic."""
    lines = []
    if relationships.get("friend"):
        lines.append(f"**Friends:** {', '.join(relationships['friend'])}")
    if relationships.get("enemy"):
        bonus = rel_config.RIVALRY_DAMAGE_BONUS_PERCENT
        lines.append(
            f"**Documented rivals:** {', '.join(relationships['enemy'])} "
            f"(canon rivalry — +{bonus}% damage vs each, per the wiki's own lore)"
        )
    if relationships.get("family"):
        lines.append(f"**Family:** {', '.join(relationships['family'])}")
    return lines


def format_weakness_report(character_name, defending_types, weak_type, rival_title, relationships=None):
    type_label = "/".join(defending_types)
    weak_to, resists, immune = [], [], []
    for attacking_type in rpg_types.POKEMON_TYPES:
        multiplier = rpg_types.combined_effectiveness(attacking_type, defending_types)
        if multiplier == 0:
            immune.append(attacking_type)
        elif multiplier > 1:
            weak_to.append(attacking_type)
        elif multiplier < 1:
            resists.append(attacking_type)

    lines = [f"**{character_name}** — {type_label} Type", ""]
    lines.append(f"Weak to (2x+ damage): {', '.join(weak_to) if weak_to else 'none'}")
    lines.append(f"Resists (0.5x or less damage): {', '.join(resists) if resists else 'none'}")
    lines.append(f"Immune to: {', '.join(immune) if immune else 'none'}")

    if weak_type:
        lines.append("")
        if rival_title:
            lines.append(
                f"**Rival:** {rival_title} — a real wiki brainrot whose {weak_type} theme "
                f"would be super effective against {character_name}."
            )
        else:
            lines.append(f"No standout {weak_type}-themed rival found in the wiki.")

    relationship_lines = format_relationships(relationships or {})
    if relationship_lines:
        lines.append("")
        lines.extend(relationship_lines)

    return "\n".join(lines)
