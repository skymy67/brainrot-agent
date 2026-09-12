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
"""

from google.genai import types
from pydantic import BaseModel

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


def format_weakness_report(character_name, defending_types, weak_type, rival_title):
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
    return "\n".join(lines)
