#!/usr/bin/env python3
"""Craft Mode: the "recipe" of real-world things conceptually combined into a brainrot character.

E.g. Tralalero Tralala = 👟👟 Sneakers (item) + 🦈 Shark (base creature).

Strictly evidence-based, in a fixed priority order — never a creative free-generation, even
though this wiki's own humor might invite one:
  1. Wiki text first — many brainrot names/descriptions directly state or strongly imply their
     components (name origin, described appearance, lore-mentioned accessories). Only a
     documented or clearly, unambiguously implied component counts.
  2. If the text doesn't establish a component and a reference image is available (reusing
     app.py's IMAGE_URL_BY_TITLE / fetch_character_image() from the visual-description feature,
     always attempted here rather than gated behind an appearance-intent check — visual
     confirmation is this mode's whole purpose, not an occasional add-on), use only what's
     actually visually confirmable in it.
  3. If neither source pins down real components, say so plainly (could_not_determine) instead
     of fabricating a plausible-sounding recipe.

Each component is also classified as "animal" (the character's base creature — stays pure lore,
no item stats, per the spec: it's part of the character's identity, not a collectible) or "item"
(an object or food component, which becomes a real collectible game item). An "item" component
gets a stat-effect category and a rarity tier — see craft_items.py for the category-to-stat
mapping and rarity thresholds, both hand-editable there. Only the item's CATEGORY and its
qualitative CLASSIFICATION need wiki/image evidence (same no-invention rule as the recipe
itself); the exact stat magnitude is a game-balance judgment call, same as RPG Mode's own stat
values. Rarity itself is never guessed by the model — it's computed deterministically in code
from the item's magnitude, effect count, and the source character's own free-lookup rarity tier
(see craft_items.py and compute_item_rarity() below), with no extra Gemini call.
"""

from google.genai import types
from pydantic import BaseModel

import craft_items
import rarity_mode
from content_policy import with_content_policy
from gemini_retry import call_with_retry

GEMINI_MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 512
MAX_OUTPUT_TOKENS = 1536
MAX_EFFECTS = 2


class StatEffect(BaseModel):
    stat: str = ""  # one of craft_items.STAT_KEYS, or craft_items.STATUS_CHANCE_STAT
    magnitude: int = 0  # raw value; clamped in code against craft_items' magnitude ranges
    # Required only when stat == STATUS_CHANCE_STAT: a short description of the effect (e.g.
    # "chance to confuse the opponent"), without the percentage — code prepends "N% ". "" for
    # every ordinary stat effect.
    status_note: str = ""


class Component(BaseModel):
    emoji: str  # repeated for multiples of the same thing, e.g. "👟👟" for two shoes
    label: str
    # Free text, not a fixed enum ("base"/"accessory"/...) — brainrot characters vary too much
    # for a rigid role list (e.g. a fusion of two equally-prominent things needs something like
    # "second base", not a forced "accessory").
    role: str
    component_type: str = "item"  # one of craft_items.COMPONENT_TYPES: "animal" or "item"
    # Everything below only applies when component_type == "item" — left blank/empty for an
    # "animal" component, which stays pure lore with no item stats.
    item_category: str = ""  # one of craft_items.ITEM_CATEGORIES
    item_name: str = ""  # a short flavor name for the item (e.g. "Speed Boots"), distinct from label
    effects: list[StatEffect] = []  # 1-2 entries; empty for an "animal" component


class CraftRecipeOutput(BaseModel):
    components: list[Component] = []
    could_not_determine: bool = False
    # Required context when could_not_determine is true — never leave a bare refusal unexplained.
    undetermined_reason: str = ""


SYSTEM_INSTRUCTION = (
    "You are a strict 'recipe' analyst for the Italian Brainrot universe: given a character, you "
    "identify the real-world objects, animals, or concepts that were conceptually combined to "
    "create it — e.g. Tralalero Tralala = sneakers + shark. You must stay strictly grounded in "
    "actual evidence and NEVER invent a component, even when the source material's humor might "
    "invite creative guessing.\n\n"
    "Follow this exact priority order:\n"
    "1. Check the wiki text context first — many brainrot names and descriptions directly state "
    "or strongly imply their components (name origin, described appearance, accessories "
    "mentioned in the lore). Only use a component that is actually documented or clearly, "
    "unambiguously implied by the text.\n"
    "2. If the text doesn't explicitly establish a component and a reference image is provided "
    "below, use ONLY what is visually confirmable in that image — the base creature/object and "
    "any distinct accessories actually depicted. Do not add anything the image doesn't actually "
    "show.\n"
    "3. If the character's components genuinely cannot be determined from the text or the image "
    "(or no image is available and the text doesn't establish them either), set "
    "could_not_determine to true and explain why in undetermined_reason — never fabricate a "
    "plausible-sounding recipe to fill the gap.\n\n"
    "Each real component gets: a short, recognizable emoji (repeated for multiples of the same "
    "thing, e.g. 👟👟 for two shoes), a short label, and a short role describing what it "
    "contributes (e.g. 'base', 'accessory', 'weapon', 'second base' for a fusion of two "
    "equally-prominent things — use whatever role word actually fits, not a fixed list).\n\n"
    "Additionally, classify each component's component_type:\n"
    "- 'animal': the character's base living creature — its identity, not a collectible. Gets "
    "no item stats.\n"
    "- 'item': any object or food component. This becomes a real collectible game item and "
    "needs the fields below.\n\n"
    "For every 'item' component, pick exactly one item_category from the fixed list given in "
    "the prompt, based on what the real object actually is:\n"
    "- footwear (shoes, boots, sandals, ...) → a Speed effect\n"
    "- weapon (blades, guns, anything sharp/offensive) → an Attack effect\n"
    "- armor (shells, helmets, anything heavy/protective) → a Defense effect\n"
    "- instrument (anything that makes sound — instruments, megaphones, sirens) → EITHER a "
    "Special Attack effect OR a status_chance effect (a percent chance to inflict a status like "
    "confusion or paralysis), whichever actually fits the specific object better\n"
    "- reflective (anything shiny/mirrored/metallic-gleaming) → a Special Defense effect\n"
    "- food (anything edible) → an HP effect, and OPTIONALLY a small second Special Defense "
    "effect if it genuinely feels like it should have one (most food items don't need one — "
    "leave effects at just the one HP entry unless there's a good reason)\n\n"
    "Give each item component: item_name (a short flavor name distinct from its label, e.g. "
    "'Speed Boots' for sneakers), and one or two StatEffect entries (a food item may have two; "
    "every other category gets exactly one). Set each effect's magnitude to a number reflecting "
    "how central/powerful this specific component seems to the character — this is a game-"
    "balance judgment call, not something that needs wiki evidence the way the component itself "
    "does. For a status_chance effect, set status_note to a short description of what it does "
    "(without a percentage — just describe the effect, e.g. 'chance to confuse the opponent')."
)
SYSTEM_INSTRUCTION = with_content_policy(SYSTEM_INSTRUCTION)


def _build_prompt(character_name, wiki_context, has_image):
    image_note = (
        "A reference image is attached below — use it per step 2 of your instructions if the "
        "text alone doesn't already establish the components."
        if has_image
        else "No reference image is available for this character — rely on the wiki text alone; "
        "if that doesn't establish the components, set could_not_determine to true."
    )
    return (
        f"Context from the Italian Brainrot wiki:\n\n"
        f"{wiki_context or '(no wiki data found for this character)'}\n\n"
        f"Character: {character_name}\n\n"
        f"{image_note}\n\n"
        f"Item categories (for 'item' components' item_category field only): "
        f"{', '.join(craft_items.ITEM_CATEGORIES)}\n\n"
        f"Determine this character's crafting recipe per your instructions."
    )


def generate_recipe(gemini_client, character_name, wiki_context, image_bytes=None, image_mime_type=None):
    """Runs the structured Gemini call (multimodal when an image was successfully fetched) and
    returns the parsed CraftRecipeOutput, or None if the response didn't parse against the
    schema at all — genuinely rare with schema-enforced output, but not impossible."""
    prompt_text = _build_prompt(character_name, wiki_context, has_image=bool(image_bytes))
    if image_bytes:
        contents = types.Content(
            parts=[
                types.Part.from_text(text=prompt_text),
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
            ]
        )
    else:
        contents = prompt_text

    response = call_with_retry(lambda: gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=CraftRecipeOutput,
        ),
    ))
    return response.parsed


def _match_from_list(value, valid_values, default):
    """Case-insensitive match against a fixed vocabulary, falling back to a safe default rather
    than trusting a hallucinated value straight through — same pattern rpg_mode.py uses for
    Pokémon Types."""
    lookup = {v.lower(): v for v in valid_values}
    return lookup.get((value or "").strip().lower(), default)


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _validate_effect(raw: StatEffect, valid_stats, magnitude_range, default_stat) -> StatEffect:
    stat = _match_from_list(raw.stat, valid_stats, default_stat)
    low, high = magnitude_range
    magnitude = _clamp(raw.magnitude, low, high, low)
    status_note = ""
    if stat == craft_items.STATUS_CHANCE_STAT:
        status_note = (raw.status_note or "").strip() or "chance to inflict a status effect"
    return StatEffect(stat=stat, magnitude=magnitude, status_note=status_note)


def _validate_component(raw: Component) -> Component:
    """Defense in depth: every field the model returned is re-matched against its own fixed
    vocabulary or numeric range before being trusted — same pattern rpg_mode.py's
    _validate_battle_ability uses. An 'animal' component is returned with all item fields
    cleared; an 'item' component gets its category, item_name, and 1-2 effects fully
    re-validated."""
    emoji = (raw.emoji or "").strip()
    label = (raw.label or "Unknown Component").strip()
    role = (raw.role or "").strip()

    component_type = _match_from_list(raw.component_type, craft_items.COMPONENT_TYPES, "item")
    if component_type == "animal":
        return Component(emoji=emoji, label=label, role=role, component_type="animal")

    item_category = _match_from_list(raw.item_category, craft_items.ITEM_CATEGORIES, craft_items.FALLBACK_CATEGORY)
    valid_stats = craft_items.CATEGORY_PRIMARY_STATS.get(item_category, craft_items.STAT_KEYS)
    primary_range = craft_items.MAGNITUDE_RANGES.get(item_category, (1, 20))

    raw_effects = raw.effects[:MAX_EFFECTS]
    if raw_effects:
        primary = _validate_effect(raw_effects[0], valid_stats, primary_range, valid_stats[0])
    else:
        primary = StatEffect(stat=valid_stats[0], magnitude=primary_range[0], status_note="")
    effects = [primary]

    secondary_stat = craft_items.SECONDARY_STAT_BY_CATEGORY.get(item_category)
    if secondary_stat and len(raw_effects) >= 2:
        secondary_raw = raw_effects[1]
        magnitude = _clamp(secondary_raw.magnitude, *craft_items.SECONDARY_MAGNITUDE_RANGE, craft_items.SECONDARY_MAGNITUDE_RANGE[0])
        effects.append(StatEffect(stat=secondary_stat, magnitude=magnitude, status_note=""))

    item_name = (raw.item_name or "").strip() or label

    return Component(
        emoji=emoji, label=label, role=role, component_type="item",
        item_category=item_category, item_name=item_name, effects=effects,
    )


def _magnitude_points(item_category, magnitude):
    low, high = craft_items.MAGNITUDE_RANGES.get(item_category, (1, 20))
    magnitude = max(low, min(high, magnitude))
    fraction = (magnitude - low) / (high - low) if high > low else 1.0
    for threshold, points in craft_items.MAGNITUDE_POINT_BUCKETS:
        if fraction <= threshold:
            return points
    return craft_items.MAGNITUDE_POINT_BUCKETS[-1][1]


def compute_item_rarity(character_name, item_category, primary_magnitude, effect_count):
    """Deterministic, code-only tier assignment for one item — no extra Gemini call. Mirrors
    rarity_mode.py's own OG-short-circuits-before-scoring structure: an OG character's own items
    are all OG, regardless of the item's own stats. Otherwise scores magnitude (normalized
    against that category's own range), effect count, and the character's own free-lookup
    rarity tier (never its search-grounded guess) against craft_items.RARITY_SCORE_THRESHOLDS."""
    if rarity_mode.check_hardcoded_og(character_name):
        return "OG"

    known_tier = rarity_mode.check_known_game_rarity(character_name)
    character_bonus = craft_items.CHARACTER_TIER_BONUS.get(known_tier, 0)

    score = (
        _magnitude_points(item_category, primary_magnitude)
        + craft_items.EFFECT_COUNT_BONUS.get(effect_count, 0)
        + character_bonus
    )
    for min_score, requires_multi, tier in craft_items.RARITY_SCORE_THRESHOLDS:
        if score >= min_score and (not requires_multi or effect_count >= 2):
            return tier
    return "Common"


def _format_effect(effect: StatEffect):
    if effect.stat == craft_items.STATUS_CHANCE_STAT:
        return f"{effect.magnitude}% {effect.status_note}".strip()
    display = craft_items.STAT_DISPLAY_NAMES.get(effect.stat, effect.stat.replace("_", " ").title())
    return f"+{effect.magnitude} {display}"


def _format_component_line(component: Component, character_name):
    if component.component_type == "animal":
        return f"{component.emoji} {component.label} (base creature — not an item)"

    effects_text = ", ".join(_format_effect(effect) for effect in component.effects) or "no effect"
    primary_magnitude = component.effects[0].magnitude if component.effects else 0
    rarity = compute_item_rarity(character_name, component.item_category, primary_magnitude, len(component.effects))
    return f"{component.emoji} {component.label} (item) — {component.item_name}, {effects_text} [{rarity}]"


def format_recipe(character_name, recipe: CraftRecipeOutput):
    if recipe.could_not_determine or not recipe.components:
        reason = recipe.undetermined_reason.strip() or "not enough evidence in the wiki text or a reference image."
        return f"{character_name}: couldn't determine its recipe — {reason}"

    lines = [f"{character_name} recipe:"]
    for raw_component in recipe.components:
        component = _validate_component(raw_component)
        lines.append(_format_component_line(component, character_name))
    return "\n".join(lines)


def build_recipe(gemini_client, character_name, wiki_context, image_bytes=None, image_mime_type=None):
    recipe = generate_recipe(gemini_client, character_name, wiki_context, image_bytes, image_mime_type)
    if recipe is None:
        return f"Couldn't generate a recipe for {character_name} — try asking again."
    return format_recipe(character_name, recipe)
