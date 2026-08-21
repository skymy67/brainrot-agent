#!/usr/bin/env python3
"""Rarity Mode: estimates a canonical rarity tier for a brainrot character.

Pipeline:
  1. OG check (hardcoded list, then LLM/search fallback) — short-circuits everything else.
  2. Wiki RAG context is gathered by the caller (app.py) and passed in.
  3+4. A single search-grounded Gemini call resolves OG status (fallback) and existing
       in-game rarity together, to minimize API calls against a tight quota.
  5+6. A second Gemini call (optionally multimodal, if the user uploaded an image) applies
       the explicit scoring rubric and produces the final tier, explanation, and signal
       breakdown.

Each non-hardcoded-OG request costs at most 2 Gemini calls.
"""

import json

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

# --- Config: easy to extend later ---------------------------------------------------

# Hardcoded OG brainrots — predate the AI-brainrot era. Case-insensitive exact match
# against the character name. Sourced from the Steal a Brainrot Wiki's own OG category
# (first-party, not a third-party blog list — those proved unreliable during review).
# Add more names here as needed.
OG_CHARACTERS = {
    "strawberry elephant",
    "meowl",
    "smurf cat",
    "skibidi toilet",
    "john pork",
    "headless horseman",
    "spyder elephant",
}

# Known "Steal a Brainrot" in-game rarities. Checked before the search-grounded lookup —
# a hit here costs zero API calls and is more reliable than an LLM web search on a topic
# this niche. Base list pulled from community guides, then corrected against user-verified
# in-game knowledge: two "Lucky Block" entries were removed (game mechanics the source
# mistakenly extracted as characters, not real brainrots), Tung Tung Tung Sahur was moved
# from Secret to Rare (the source had his post-nerf/original tier, not his current one),
# and Griffin was added to Secret (missing from the source; it had mislabeled him as OG).
# Case-insensitive exact match. Not exhaustive — anything not listed here falls through to
# the search-grounded lookup instead of being treated as "confirmed not found".
KNOWN_GAME_RARITIES = {
    # Common
    "noobini pizzanini": "Common",
    "lirilì larilà": "Common",
    "tim cheese": "Common",
    "fluriflura": "Common",
    "talpa di fero": "Common",
    "noobini santanini": "Common",
    "svinina bombardino": "Common",
    "raccooni jandelini": "Common",
    "pipi kiwi": "Common",
    "tartaragno": "Common",
    "pipi corni": "Common",
    "holy arepa": "Common",
    # Rare
    "trippi troppi": "Rare",
    "gangster footera": "Rare",
    "bandito bobritto": "Rare",
    "boneca ambalabu": "Rare",
    "cacto hipopotamo": "Rare",
    "ta ta ta ta sahur": "Rare",
    "cupcake koala": "Rare",
    "tric trac baraboom": "Rare",
    "frogo elfo": "Rare",
    "pipi avocado": "Rare",
    "pengolino nuvoletto": "Rare",
    "pinealotto fruttarino": "Rare",
    "tung tung tung sahur": "Rare",
    # Epic
    "cappuccino assassino": "Epic",
    "brr brr patapim": "Epic",
    "avocadini antilopini": "Epic",
    "trulimero trulicina": "Epic",
    "bambini crostini": "Epic",
    "malame amarele": "Epic",
    "bananita dolphinita": "Epic",
    "perochello lemonchello": "Epic",
    "brri brri bicus dicus bombicus": "Epic",
    "avocadini guffo": "Epic",
    "ti ti ti sahur": "Epic",
    "mangolini parrocini": "Epic",
    "frogato pirato": "Epic",
    "gato celesto": "Epic",
    "salamino penguino": "Epic",
    "doi doi do": "Epic",
    "penguin tree": "Epic",
    "wombo rollo": "Epic",
    "penguino cocosino": "Epic",
    "mummio rappitto": "Epic",
    # Legendary
    "chimpanzini bananini": "Legendary",
    "tirilikalika tirilikalako": "Legendary",
    "ballerina cappuccina": "Legendary",
    "burbaloni loliloli": "Legendary",
    "chef crabracadabra": "Legendary",
    "lionel cactuseli": "Legendary",
    "glorbo fruttodrillo": "Legendary",
    "quivioli ameleonni": "Legendary",
    "clickerino crabo": "Legendary",
    "blueberrinni octopusini": "Legendary",
    "caramello filtrello": "Legendary",
    "pipi potato": "Legendary",
    "strawberrelli flamingelli": "Legendary",
    "cocosini mama": "Legendary",
    "bandito axolito": "Legendary",
    "pandaccini bananini": "Legendary",
    "quackula": "Legendary",
    "pi pi watermelon": "Legendary",
    "buho del cielo": "Legendary",
    "sigma boy": "Legendary",
    "chocco bunny": "Legendary",
    "puffaball": "Legendary",
    "sigma girl": "Legendary",
    "sealo regalo": "Legendary",
    "electro quacko": "Legendary",
    "buho de fuego": "Legendary",
    "seraphino gruyero": "Legendary",
    # Mythic
    "frigo camelo": "Mythic",
    "orangutini ananassini": "Mythic",
    "rhino toasterino": "Mythic",
    "bombardiro crocodilo": "Mythic",
    "brutto gialutto": "Mythic",
    "spioniro golubiro": "Mythic",
    "bombombini gusini": "Mythic",
    "zibra zubra zibralini": "Mythic",
    "tigrilini watermelini": "Mythic",
    "avocadorilla": "Mythic",
    "cavallo virtuoso": "Mythic",
    "gorillo subwoofero": "Mythic",
    "gorillo watermelondrillo": "Mythic",
    "stoppo luminino": "Mythic",
    "ganganzelli trulala": "Mythic",
    "tob tobi tobi": "Mythic",
    "lerulerulerule": "Mythic",
    "te te te sahur": "Mythic",
    "rhino helicopterino": "Mythic",
    "magi ribbitini": "Mythic",
    "tracoducotulu delapeladustuz": "Mythic",
    "jingle jingle sahur": "Mythic",
    "los noobinis": "Mythic",
    "spongini quackini": "Mythic",
    "cachorrito melonito": "Mythic",
    "bee loco": "Mythic",
    "carloo": "Mythic",
    "harpuccino": "Mythic",
    "cocoteddy": "Mythic",
    "carrotini brainini": "Mythic",
    "centrucci nuclucci": "Mythic",
    "toiletto focaccino": "Mythic",
    "jacko spaventosa": "Mythic",
    "bananito bandito": "Mythic",
    "tree tree tree sahur": "Mythic",
    "fizzy soda": "Mythic",
    "berenjello angello": "Mythic",
    "bucketoro": "Mythic",
    "orbi mochi": "Mythic",
    "tic tic ribbit": "Mythic",
    # Brainrot God
    "cocofanto elefanto": "Brainrot God",
    "girafa celestre": "Brainrot God",
    "gattatino neonino": "Brainrot God",
    "gattatino nyanino": "Brainrot God",
    "chihuanini taconini": "Brainrot God",
    "matteo": "Brainrot God",
    "tralalero tralala": "Brainrot God",
    "los crocodillitos": "Brainrot God",
    "tigroligre frutonni": "Brainrot God",
    "odin din din dun": "Brainrot God",
    "money money man": "Brainrot God",
    "alessio": "Brainrot God",
    "statutino libertino": "Brainrot God",
    "tipi topi taco": "Brainrot God",
    "unclito samito": "Brainrot God",
    "tralalita tralala": "Brainrot God",
    "tukanno bananno": "Brainrot God",
    "extinct ballerina": "Brainrot God",
    "vampira cappucina": "Brainrot God",
    "espresso signora": "Brainrot God",
    "orcalero orcala": "Brainrot God",
    "trenostruzzo turbo 3000": "Brainrot God",
    "jacko jack jack": "Brainrot God",
    "urubini flamenguini": "Brainrot God",
    "trippi troppi troppa trippa": "Brainrot God",
    "capi taco": "Brainrot God",
    "sundrilla sundae": "Brainrot God",
    "divino platypio": "Brainrot God",
    "los chihuaninis": "Brainrot God",
    "gattito tacoto": "Brainrot God",
    "las capuchinas": "Brainrot God",
    "pineaplino": "Brainrot God",
    "bulbito bandito traktorito": "Brainrot God",
    "ballerino lololo": "Brainrot God",
    "los tungtungtungcitos": "Brainrot God",
    "ballerina peppermintina": "Brainrot God",
    "pakrahmatmamat": "Brainrot God",
    "brr es teh patipum": "Brainrot God",
    "piccione macchina": "Brainrot God",
    "pakrahmatmatina": "Brainrot God",
    "los bombinitos": "Brainrot God",
    "tractoro dinosauro": "Brainrot God",
    "los orcalitos": "Brainrot God",
    "cacasito satalito": "Brainrot God",
    "orcalita orcala": "Brainrot God",
    "corn corn corn sahur": "Brainrot God",
    "mummy ambalabu": "Brainrot God",
    "snailenzo": "Brainrot God",
    "squalanana": "Brainrot God",
    "tartaruga cisterna": "Brainrot God",
    "aquanaut": "Brainrot God",
    "trenoturbo axolotrico 9000": "Brainrot God",
    "lazy ducky": "Brainrot God",
    "ginger globo": "Brainrot God",
    "yeti claus": "Brainrot God",
    "crabbo limonetta": "Brainrot God",
    "tootini shrimpini": "Brainrot God",
    "granchiello spiritell": "Brainrot God",
    "los tipi tacos": "Brainrot God",
    "frio ninja": "Brainrot God",
    "quackalena": "Brainrot God",
    "lumaca malefica": "Brainrot God",
    "buho de noelo": "Brainrot God",
    "bunny tralala": "Brainrot God",
    "boba panda": "Brainrot God",
    "piccionetta macchina": "Brainrot God",
    "cola cat": "Brainrot God",
    "bambu bambu sahur": "Brainrot God",
    "los gattitos": "Brainrot God",
    "mastodontico telepiedone": "Brainrot God",
    "chrismasmamat": "Brainrot God",
    "lemonita splashita": "Brainrot God",
    "astrolero cervalero": "Brainrot God",
    "anpali babel": "Brainrot God",
    "luv luv luv": "Brainrot God",
    "cappuccino clownino": "Brainrot God",
    "bombardini tortinii": "Brainrot God",
    "brasilini berimbini": "Brainrot God",
    "patteo": "Brainrot God",
    "belula beluga": "Brainrot God",
    "krupuk pagi pagi": "Brainrot God",
    "skull skull skull": "Brainrot God",
    "cocoa assassino": "Brainrot God",
    "tentacolo tecnico": "Brainrot God",
    "ginger cisterna": "Brainrot God",
    "pandanini frostini": "Brainrot God",
    "dolphini jetskini": "Brainrot God",
    "pop pop sahur": "Brainrot God",
    "noo la polizia": "Brainrot God",
    "karkerheart luvkur": "Brainrot God",
    "appelini": "Brainrot God",
    "clovkur kurkur": "Brainrot God",
    "eggdin egg egg dun": "Brainrot God",
    "dumborino miracello": "Brainrot God",
    "flippo marino": "Brainrot God",
    "robo grafito": "Brainrot God",
    "tortuginni sandcastlini": "Brainrot God",
    "tenini ballini": "Brainrot God",
    # Secret
    "la vacca saturno saturnita": "Secret",
    "bisonte giuppitere": "Secret",
    "sammyni spyderini": "Secret",
    "blackhole goat": "Secret",
    "pretzo robo": "Secret",
    "jackorilla": "Secret",
    "karkerkar kurkur": "Secret",
    "agarrini la palini": "Secret",
    "chachechi": "Secret",
    "trenostruzzo turbo 4000": "Secret",
    "chimpanzini spiderini": "Secret",
    "los matteos": "Secret",
    "los tortus": "Secret",
    "los tralaleritos": "Secret",
    "la cucaracha": "Secret",
    "vulturino skeletono": "Secret",
    "boatito auratito": "Secret",
    "torrtuginni dragonfrutini": "Secret",
    "los spyderinis": "Secret",
    "extinct tralalero": "Secret",
    "fragola la la la": "Secret",
    "zombie tralala": "Secret",
    "yess my examen": "Secret",
    "extinct matteo": "Secret",
    "dul dul dul": "Secret",
    "guerriro digitale": "Secret",
    "las tralaleritas": "Secret",
    "rocco disco": "Secret",
    "la karkerkar combinasion": "Secret",
    "la vacca prese presente": "Secret",
    "reindeer tralala": "Secret",
    "pumpkini spyderini": "Secret",
    "los trios": "Secret",
    "frankentteo": "Secret",
    "job job job sahur": "Secret",
    "karker sahur": "Secret",
    "las vaquitas saturnitas": "Secret",
    "los karkeritos": "Secret",
    "santteo": "Secret",
    "fishboard": "Secret",
    "buntteo": "Secret",
    "la vacca jacko linterino": "Secret",
    "triplito tralaleritos": "Secret",
    "paradiso axolottino": "Secret",
    "trickolino": "Secret",
    "goat": "Secret",
    "giftini spyderini": "Secret",
    "love love love sahur": "Secret",
    "graipuss medussi": "Secret",
    "perrito burrito": "Secret",
    "bombardiro vaccariro": "Secret",
    "la vacca lepre lepreino": "Secret",
    "1x1x1x1": "Secret",
    "easter easter easter sahur": "Secret",
    "los cucarachas": "Secret",
    "hippo golazo": "Secret",
    "please my present": "Secret",
    "craburger": "Secret",
    "gelato lumacho": "Secret",
    "cuadramat and pakrahmatmamat": "Secret",
    "berryno": "Secret",
    "bunnyman": "Secret",
    "coffin tung tung tung sahur": "Secret",
    "los jobcitos": "Secret",
    "nooo my hotspot": "Secret",
    "la sahur combinasion": "Secret",
    "list list list sahur": "Secret",
    "griffin": "Secret",
}

# Ordered lowest to highest. OG is intentionally excluded — it's a separate, non-power-based
# flag handled outside this list.
RARITY_TIERS = [
    "Common",
    "Uncommon",
    "Rare",
    "Epic",
    "Legendary",
    "Mythic",
    "Brainrot God",
    "Secret",
    "Celestial",
]

# Each of the 5 rubric categories (existing game rarity, size, weight, lore, battle history)
# scores 0/1/2, for a max total of 10. Mid tiers are pure score thresholds; the top two tiers
# additionally require signal spread across categories (not one maxed-out category carrying
# the whole score) per the spec's "reserve Secret/Celestial for exceptional signals across
# multiple categories" requirement.
SCORE_THRESHOLDS = [
    # (min_score, max_score, min_categories_maxed, tier)
    (0, 0, 0, "Common"),
    (1, 2, 0, "Uncommon"),
    (3, 4, 0, "Rare"),
    (5, 6, 0, "Epic"),
    (7, 8, 0, "Legendary"),  # score 7-8 without enough spread stays Legendary
    (8, 8, 3, "Mythic"),  # score 8 with >=3 maxed categories
    (9, 9, 3, "Brainrot God"),
    (10, 10, 4, "Secret"),
    (10, 10, 5, "Celestial"),  # perfect score, every category maxed
]

GEMINI_MODEL = "gemini-3.6-flash"
SEARCH_THINKING_BUDGET = 512
SEARCH_MAX_OUTPUT_TOKENS = 2048
SCORE_THINKING_BUDGET = 768
SCORE_MAX_OUTPUT_TOKENS = 3072


# --- Structured output schemas --------------------------------------------------------


class OgAndRaritySearchResult(BaseModel):
    is_og: bool
    og_reasoning: str
    existing_game_rarity_found: bool
    existing_game_rarity: str  # e.g. "Mythic in Steal a Brainrot", or "" if not found
    existing_game_rarity_reasoning: str


class RarityScoreResult(BaseModel):
    tier: str
    total_score: int
    explanation: str
    size_summary: str
    weight_summary: str
    lore_summary: str
    battle_summary: str
    existing_game_rarity_summary: str
    categories_maxed: int


# --- OG check ---------------------------------------------------------------------------


def check_hardcoded_og(character_name):
    return character_name.strip().lower() in OG_CHARACTERS


def check_known_game_rarity(character_name):
    return KNOWN_GAME_RARITIES.get(character_name.strip().lower())


def score_to_tier(total_score, categories_maxed):
    total_score = max(0, min(10, total_score))
    best_match = "Common"
    for min_score, max_score, min_categories_maxed, tier in SCORE_THRESHOLDS:
        if min_score <= total_score <= max_score and categories_maxed >= min_categories_maxed:
            best_match = tier
    return best_match


# --- Gemini calls -------------------------------------------------------------------------


def run_og_and_rarity_search(gemini_client: "genai.Client", character_name, wiki_context):
    system_instruction = (
        "You are a research assistant scoped strictly to the 'brainrot' internet meme trend "
        "and brainrot-themed games (Steal a Brainrot and similar). You may ONLY search for "
        "brainrot-related terms — always include the word 'brainrot' or a specific brainrot "
        "game name in any search query you issue. Refuse to search for anything outside this "
        "scope, even if asked. You have two jobs: (1) determine whether the given character "
        "predates the AI-generated 'Italian brainrot' trend (roughly early 2025 onward) — an "
        "'OG' meme/character that existed before brainrot content used it, as opposed to a "
        "character invented as part of the brainrot trend itself; (2) determine whether the "
        "character has an existing rarity tier in 'Steal a Brainrot' or a similar brainrot "
        "collector game."
    )
    # Structured output (response_schema) isn't supported together with the google_search
    # tool — when tools are enabled, the model's output is driven by its tool-calling flow,
    # not schema enforcement. So we ask for JSON in the prompt instead and parse it manually.
    prompt = (
        f"Character: {character_name}\n\n"
        f"Wiki context (may be empty if not found):\n{wiki_context or '(no wiki data found for this character)'}\n\n"
        f"Search for 'brainrot {character_name} Steal a Brainrot rarity' and similar "
        f"brainrot-scoped queries to answer both questions. If you can't find an existing "
        f"in-game rarity, set existing_game_rarity_found to false — do not guess or default "
        f"to a random tier.\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no other text) matching this "
        f"exact shape:\n"
        f'{{"is_og": bool, "og_reasoning": str, "existing_game_rarity_found": bool, '
        f'"existing_game_rarity": str, "existing_game_rarity_reasoning": str}}'
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=types.ThinkingConfig(thinking_budget=SEARCH_THINKING_BUDGET),
            max_output_tokens=SEARCH_MAX_OUTPUT_TOKENS,
        ),
    )
    return parse_og_and_rarity_json(response.text)


def parse_og_and_rarity_json(raw_text):
    """Best-effort parse of Step A's freeform JSON. Falls back to a safe default (not OG,
    no existing rarity found) if the model didn't return clean JSON, so a parsing hiccup
    degrades gracefully instead of crashing the pipeline."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text.strip())
        return OgAndRaritySearchResult(**data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return OgAndRaritySearchResult(
            is_og=False,
            og_reasoning="Could not determine OG status (search step returned an unparseable response).",
            existing_game_rarity_found=False,
            existing_game_rarity="",
            existing_game_rarity_reasoning="Could not parse search results.",
        )


def run_final_scoring(gemini_client: "genai.Client", character_name, wiki_context, search_result, image_bytes, image_mime_type):
    rubric = (
        "Score the character on 5 categories, each worth 0, 1, or 2 points (max total 10):\n\n"
        "1. Existing in-game rarity: not found = 0. Maps to Common/Uncommon/Rare in-game = 0, "
        "Epic/Legendary in-game = 1, Mythic/Brainrot God/Secret/Celestial in-game = 2.\n"
        "2. Size: extreme (very large, very small, or unusual proportions) = 2, "
        "above-average = 1, average = 0.\n"
        "3. Weight: same scale as size — extreme = 2, above-average = 1, average = 0.\n"
        "4. Lore: unique/notable backstory or role in wiki lore = 2, some lore = 1, no lore = 0.\n"
        "5. Battle history: undefeated/dominant record = 2, mixed record = 1, "
        "no battle history = 0.\n\n"
        "If BOTH lore and battle history are missing (score 0 on both), compare this "
        "character's signals against similar brainrots with known game rarities and weight "
        "the size/weight/existing-rarity categories accordingly instead of leaving everything "
        "at 0.\n\n"
        "Report total_score (sum of all 5 categories, 0-10) and categories_maxed (how many "
        "of the 5 categories scored exactly 2) honestly — do not adjust these to hit a tier "
        "you think sounds right; the tier is derived from your scores programmatically."
    )

    og_search_summary = (
        f"OG check: not OG ({search_result.og_reasoning})\n"
        f"Existing game rarity: "
        f"{search_result.existing_game_rarity if search_result.existing_game_rarity_found else 'not found'} "
        f"({search_result.existing_game_rarity_reasoning})"
    )

    prompt_parts = [
        types.Part.from_text(
            text=(
                f"Character: {character_name}\n\n"
                f"Wiki context (lore, battle history, canon size/weight if present):\n"
                f"{wiki_context or '(no wiki data found for this character)'}\n\n"
                f"{og_search_summary}\n\n"
                f"{rubric}"
            )
        )
    ]

    if image_bytes:
        prompt_parts.insert(
            0,
            types.Part.from_text(
                text=(
                    "The user uploaded a reference image of this character below. Use it to "
                    "estimate height (m or mm — use your judgment on scale) and weight (kg or "
                    "g — use your judgment on scale) based on visual proportions and comparisons "
                    "to known objects/creatures, then use that estimate for the size/weight "
                    "categories in the rubric."
                )
            ),
        )
        prompt_parts.insert(1, types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=types.Content(parts=prompt_parts),
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a rarity-tier judge for the Italian Brainrot wiki, applying a strict, "
                "explicit scoring rubric consistently across characters. Never assign a tier "
                "from vibes alone — always derive it from the rubric's point totals."
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=SCORE_THINKING_BUDGET),
            max_output_tokens=SCORE_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=RarityScoreResult,
        ),
    )
    if response.parsed is None:
        return RarityScoreResult(
            tier="Common",
            total_score=0,
            explanation="Scoring failed — the model didn't return a parseable result. Defaulted to Common.",
            size_summary="unknown",
            weight_summary="unknown",
            lore_summary="unknown",
            battle_summary="unknown",
            existing_game_rarity_summary=og_search_summary,
            categories_maxed=0,
        )
    return response.parsed


# --- Output formatting ------------------------------------------------------------------


def format_og_answer(character_name, reasoning):
    return (
        f"**{character_name} — Rarity: OG**\n\n"
        f"{reasoning}\n\n"
        f"OG is a special tier reserved for characters that predate the AI-brainrot trend — "
        f"it isn't power-based, so no scoring was applied."
    )


def format_score_answer(character_name, result: RarityScoreResult):
    computed_tier = score_to_tier(result.total_score, result.categories_maxed)
    return (
        f"**{character_name} — Rarity: {computed_tier}**\n\n"
        f"{result.explanation}\n\n"
        f"**Signal breakdown** (score: {result.total_score}/10):\n"
        f"- Existing game rarity: {result.existing_game_rarity_summary}\n"
        f"- Size: {result.size_summary}\n"
        f"- Weight: {result.weight_summary}\n"
        f"- Lore: {result.lore_summary}\n"
        f"- Battle history: {result.battle_summary}"
    )


# --- Pipeline entry point ----------------------------------------------------------------


def _unavailable_search_result(reason):
    """Fallback when the search-grounded call itself fails (quota, billing restriction,
    transient outage) — distinct from a successful call that returns unparseable JSON.
    Lets the pipeline proceed to scoring instead of failing the whole request, per the
    spec's 'do not default to random, weight other signals more heavily instead' rule."""
    return OgAndRaritySearchResult(
        is_og=False,
        og_reasoning=f"Could not determine OG status ({reason}).",
        existing_game_rarity_found=False,
        existing_game_rarity="",
        existing_game_rarity_reasoning=f"Existing-rarity search unavailable ({reason}).",
    )


def run_rarity_pipeline(gemini_client, character_name, wiki_context, image_bytes=None, image_mime_type=None):
    if check_hardcoded_og(character_name):
        return format_og_answer(character_name, f"{character_name} is a known OG brainrot — its origin predates the AI-brainrot trend.")

    known_rarity = check_known_game_rarity(character_name)
    if known_rarity:
        search_result = OgAndRaritySearchResult(
            is_og=False,
            og_reasoning="Not in the OG list.",
            existing_game_rarity_found=True,
            existing_game_rarity=f"{known_rarity} in Steal a Brainrot",
            existing_game_rarity_reasoning="Matched against a locally maintained list of known in-game rarities — no search needed.",
        )
    else:
        try:
            search_result = run_og_and_rarity_search(gemini_client, character_name, wiki_context)
        except genai_errors.APIError as exc:
            search_result = _unavailable_search_result(f"API error {exc.code}")

    if search_result.is_og:
        return format_og_answer(character_name, search_result.og_reasoning)

    score_result = run_final_scoring(gemini_client, character_name, wiki_context, search_result, image_bytes, image_mime_type)
    return format_score_answer(character_name, score_result)
