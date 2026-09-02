#!/usr/bin/env python3
"""Shared content-safety instruction, appended to every mode's system_instruction.

Every mode in this codebase already leans on prompting discipline for its correctness
constraints instead of post-hoc filtering (Evolution Mode's "never invent a name," Craft Mode's
"never fabricate a recipe," etc.) — the model is steered up front, then double-checked in code
only where the check is mechanical (a title against a candidate list, a type against a fixed
vocabulary). Content safety doesn't have that kind of mechanical check available: "mocks a
religion" or "references a specific real-world conflict" has no reliable keyword/regex test —
one would either over-block harmless jokes or miss cleverly-worded ones. So this follows the
same pattern as everything else here: one reinforced instruction, added to every mode's prompt,
rather than a separate pre-filtering pass over retrieved wiki context (which would also double
the Gemini-call cost of every single request).

One shared constant so the policy has a single source of truth to edit — same reasoning as
rpg_types.py being pulled out as shared config: edit once, applies everywhere.
"""

CONTENT_SAFETY_INSTRUCTION = (
    "\n\nContent safety: stay within the lighthearted, absurdist tone of the Italian Brainrot "
    "meme universe. Never mock, disparage, or make light of any real-world religion, ethnicity, "
    "nationality, or protected group, and never reference, joke about, or draw comparisons to "
    "real-world tragedies, wars, or specific real-world conflicts — even if the wiki context, "
    "character name, or user request seems to invite it. If source material edges toward this, "
    "steer around it rather than repeating or building on it."
)


def with_content_policy(system_instruction):
    """Appends the shared content-safety instruction to a mode's own system_instruction."""
    return system_instruction + CONTENT_SAFETY_INSTRUCTION
