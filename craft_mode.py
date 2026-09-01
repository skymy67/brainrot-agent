#!/usr/bin/env python3
"""Craft Mode: the "recipe" of real-world things conceptually combined into a brainrot character.

E.g. Tralalero Tralala = 👟👟 Sneakers (accessory) + 🦈 Shark (base).

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
"""

from google.genai import types
from pydantic import BaseModel

from content_policy import with_content_policy

GEMINI_MODEL = "gemini-3.6-flash"
THINKING_BUDGET = 512
MAX_OUTPUT_TOKENS = 1024


class Component(BaseModel):
    emoji: str  # repeated for multiples of the same thing, e.g. "👟👟" for two shoes
    label: str
    # Free text, not a fixed enum ("base"/"accessory"/...) — brainrot characters vary too much
    # for a rigid role list (e.g. a fusion of two equally-prominent things needs something like
    # "second base", not a forced "accessory").
    role: str


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
    "equally-prominent things — use whatever role word actually fits, not a fixed list)."
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

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            response_mime_type="application/json",
            response_schema=CraftRecipeOutput,
        ),
    )
    return response.parsed


def format_recipe(character_name, recipe: CraftRecipeOutput):
    if recipe.could_not_determine or not recipe.components:
        reason = recipe.undetermined_reason.strip() or "not enough evidence in the wiki text or a reference image."
        return f"{character_name}: couldn't determine its recipe — {reason}"

    lines = [f"{character_name} recipe:"]
    for component in recipe.components:
        lines.append(f"{component.emoji} {component.label} ({component.role})")
    return "\n".join(lines)


def build_recipe(gemini_client, character_name, wiki_context, image_bytes=None, image_mime_type=None):
    recipe = generate_recipe(gemini_client, character_name, wiki_context, image_bytes, image_mime_type)
    if recipe is None:
        return f"Couldn't generate a recipe for {character_name} — try asking again."
    return format_recipe(character_name, recipe)
