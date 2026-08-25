#!/usr/bin/env python3
"""Akinator Mode: a 20-questions-style guessing game restricted to real wiki characters.

Every question topic and every guess is chosen from the actual set of characters in
wiki_data.json — the model is never asked to invent one. To make that tractable without
feeding thousands of full wiki pages to Gemini on every turn, candidate narrowing is driven by
each page's own MediaWiki `Category:` tags (mined once at startup, the same first-party signal
already validated for Evolution Mode) rather than free-form reasoning. Gemini's role is
narrower than "solve the game" — it phrases a chosen trait into a natural question, and once
the leaderboard has narrowed to a small shortlist, picks the best-fitting final guess from it.

Narrowing itself is a weighted-score model, not a hard filter — matching the real Akinator's
five-answer scale (Yes / Probably / Don't Know / Probably Not / No) rather than a strict
yes/no/unsure. Every candidate keeps a running plausibility score for the whole round instead
of being definitively kept or eliminated; each answer nudges matching candidates one way and
non-matching candidates the other, by an amount that scales with how confident the answer was.
This is deliberately not "closest-to-50% pool elimination": a single too-strict "no" on a trait
the player is genuinely unsure about can't permanently kill the correct candidate the way hard
filtering does, and a hedged "probably"/"probably not" carries real but smaller weight than a
committed "yes"/"no". Question *selection* still greedily picks whichever available tag splits
the current top-scoring candidates closest to 50/50 — the same "most informative next question"
heuristic — just evaluated against the leaderboard instead of a shrinking eligible set.

Session state (per-candidate scores, asked tags, question count, round history) has nowhere to
live server-side in this app — there's no database or session store, and everything else the
backend does is a single stateless request/response. So state round-trips through the
request/response bodies each turn, the same way the frontend already persists chat history to
sessionStorage; the caller (app.py) is only responsible for handing back whatever dict this
module last returned.

Tag *weights*, unlike round state, do persist server-side — a small on-disk store (see "Learned
tag reliability" below) accumulates across every round ever revealed, on top of DATA_DIR (a
Railway volume in production). This is the piece the wiki-derived, static tag data alone can't
provide: real Akinator's own accuracy comes largely from collaborative filtering over millions
of aggregated player answers, not just a fixed trait table. Here, every revealed round checks
each answered question against the real character's actual tags and records whether the tag
"held up" — a tag that keeps proving unreliable across many rounds (players routinely answer it
in a way that turns out to contradict the revealed character) gets down-weighted in future
scoring, and a consistently reliable one gets amplified, without ever needing new wiki data.
"""

import json
import os
import re
import threading
from typing import Literal

from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

# Points at a Railway volume in production (set via the DATA_DIR env var) so the learned tag
# stats below survive redeploys; defaults to the working directory for local dev, same as
# wiki_data.json/chroma_db already implicitly do.
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)
LEARNING_FILE = os.path.join(DATA_DIR, "akinator_learning.json")

WIKI_DATA_FILE = "wiki_data.json"
GEMINI_MODEL = "gemini-3.6-flash"
MAX_QUESTIONS = 30
MAX_SHORTLIST_FOR_GEMINI = 8
CONTENT_TRUNCATE_CHARS = 500
QUESTION_THINKING_BUDGET = 256
QUESTION_MAX_OUTPUT_TOKENS = 256
GUESS_THINKING_BUDGET = 512
GUESS_MAX_OUTPUT_TOKENS = 512

# Category tags can be multi-word ("Category:Not Italian", "Category:Sahur family"), and on
# ~31 of the wiki's 4687 pages, multiple categories appear back-to-back on the same source line
# with no newline between them at all (a mwparserfromhell plain-text rendering quirk). Matching
# only up to \n (or only up to the next literal newline) either truncates multi-word tags to
# their first word, or — worse — swallows every subsequent "Category:X" on that line into one
# giant garbage "tag" (e.g. "Category:Ostrich Category:Toilet" with no separator became the
# single fake tag "OstrichCategory:ToiletCategory:Train"). That silently dropped real tags for
# those pages, degrading how well the pool actually narrows for them. Stopping the match at the
# next "Category:" occurrence too (not just \n) splits these correctly.
CATEGORY_RE = re.compile(r"Category:(.*?)(?=Category:|\n|$)")

# Tags that exist on real wiki pages but aren't player-observable traits of the character
# itself — production/attribution metadata (which AI tool or human made the page), wiki
# housekeeping/editorial status, release-date or meme-age tracking, or anything tier/power/
# popularity-flavored (explicitly out of scope per the spec: rarity-tier and stat-like data
# isn't fair question material here, RPG/Rarity mode already cover that). A player thinking of
# a character has no way to know these about their character's wiki page, so a tag like this
# getting picked as a question forces an "unsure" answer that narrows nothing and wastes a turn.
# Identified by inspecting the full ~900-tag frequency table (not just the top ones) for
# editorial/attribution/meta patterns; an obscure one attached to only a page or two may still
# slip through uncaught, an acceptable, disclosed limitation rather than a correctness bug.
EXCLUDED_TAGS = {
    "Wikimades",
    "Wikimade",
    "Legacy DALL-E",
    "Gemini",
    "Non-AI",
    "Characters",
    "Italian Brainrot Characters",
    "Famous",
    "Popular in Roblox",
    "Strongest",
    "Alexey Pigeon",
    "Noxa",
    "Articles with potentially offensive material",
    "Italian brainrot creators",
    "AI Reviewed Pages",
    "Candidates for Shining Articles",
    "Shining article",
    "Featured",
    "Stubs",
    "Long articles",
    "Disambiguation",
    "Non-Mainstream Content",
    "Non-mainstream content",
    "Unverified content",
    "Uncategorized Pages",
    "Formerly Deleted",
    "Deleted",
    "Old Wiki",
    "Undeleted pages",
    "Steal a Brainrot-originating",
    "Craft a Brainrot Exclusive",
    "Very Young Brainrots",
    "Non-existent Brainrot",
    "Non-Brainrots",
    "Bing DALL-E",
    "Dreamina AI",
    # These looked at first like in-universe fictional families ("is your character part of the
    # X group?"), until checking their actual member pages: each groups together completely
    # unrelated character names/themes with no shared trait, the same pattern as a wiki
    # contributor's own username getting auto-attached to every page they created — e.g.
    # "Beldarian" tags "Vinny", "William", and "Barrette" with nothing in common. A player has no
    # way to know who made their character's page, so these are attribution metadata like
    # "Wikimades" above, not player-observable traits. Found while building SUPERCATEGORIES
    # below and cross-checking every candidate family tag's actual member pages for coherence.
    "Karkerkur",
    "Modmades",
    "Doombasterd",
    "Beldarian",
    "Bicicleteira",
    "Viciosini",
    "Yibaty",
    "Saphiri",
    "Larila",
    "Cani-mungos",
    "Picciones",
    "Goaat Galaxy",
    "Breno",
    # In-game/collector rarity tiers — same "Steal a Brainrot" tier system Rarity Mode already
    # covers, explicitly out of scope per the spec alongside the numeric "Tier X-Y" tags below.
    "Common",
    "Uncommon",
    "Rare",
    "Epic",
    "Legendary",
    "Mythic",
    "Brainrot God",
    "Secret",
    "Celestial",
    "OG",
}
EXCLUDED_TAG_PATTERNS = [
    re.compile(r"^Tier\b", re.IGNORECASE),  # power-tier classifications, e.g. "Tier 10-C"
    re.compile(r"'s [Bb]rainrots$"),  # creator-attribution tags, e.g. "Henzwxz's Brainrots"
    re.compile(r"^Brainrots of \w+ \d{4}$"),  # release-period tags, e.g. "Brainrots of March 2025"
    re.compile(r"^\w+ \d{4} Brainrots$"),  # release-period tags, other word order
    re.compile(r"\d Years?.? Old Brainrots$", re.IGNORECASE),  # meme-age tags
    re.compile(r"\bAi Video Maker\b", re.IGNORECASE),  # AI-generation-tool attribution
]


def _is_excluded_tag(tag):
    if tag in EXCLUDED_TAGS:
        return True
    return any(pattern.search(tag) for pattern in EXCLUDED_TAG_PATTERNS)


# The wiki's own Category: tags are extremely long-tailed — over 900 unique tags across 4687
# pages, and even the single most common one ("Not Italian") only covers ~6% of the wiki. A
# player who answers "no" to several of the most common individual tags in a row (the normal
# case for any character that isn't one of a handful of very common types) barely narrows the
# pool at all: a real test round for "Cocofanto Elefanto" (tagged Elephants/Coconut/Jungle,
# none of which are common) went 4687 -> 3659 candidates over the full 30-question budget and
# never converged. Grouping related low-frequency tags into broad, genuinely intuitive
# supercategories (the kind of question a player would expect — "is it an animal?" — matching
# direct user feedback) gives real, meaningfully bigger splits: "Animal" alone covers ~12% of
# the wiki versus ~6% for the best single raw tag, and more importantly, an animal-themed
# character now answers "yes" to ONE broad early question instead of "no" to a dozen unrelated
# ones before its own specific tags ever become common enough to be picked.
#
# This started as 4 very broad groups (Animal/Food/Object/Setting). It's since been expanded to
# ~50 categories at two tiers: the original 4 broad ones for the earliest, biggest splits, plus
# narrower sub-groups (e.g. "Feline"/"Bird"/"Fruit"/"Vehicle") for once the pool has narrowed
# enough that the broad category alone no longer discriminates well, plus a handful of specific
# named in-universe families (e.g. "Sahur Family", "67 Family") that are genuine, well-defined
# identity groups worth asking about directly. No question-selection code changes were needed
# for this — _best_split_tag() already treats every tag (raw or supercategory, broad or narrow)
# as one undifferentiated pool and just picks whichever currently splits the leaders closest to
# 50/50, so adding more candidate tags here only gives it better options to choose from.
#
# Built by inspecting the full ~900-tag frequency table (same process PR #17's EXCLUDED_TAGS
# audit used) and grouping tags with a genuinely shared, player-observable theme. Several
# small candidate "families" were checked against their actual member pages and dropped for
# incoherence (e.g. a 1-member "family" isn't a useful question) or turned out to be wiki
# contributor usernames masquerading as categories (see the EXCLUDED_TAGS additions above).
# Categories intentionally don't need to be mutually exclusive — a character can and often does
# belong to several (e.g. a cat character matches both "Animal" and "Feline"), same as before.
# Each candidate's tag set gets every matching supercategory name added in addition to (not
# instead of) its own specific tags, so finer distinctions are still available once the pool
# narrows.
SUPERCATEGORIES = {
    "Animal": {
        "Animals", "Antelope", "Aquatic", "Axolotl", "Bat", "Bears", "Beaver", "Beavers",
        "Big Cats", "Bird", "Birds", "Buffalo", "Butterfly", "Camels", "Capybara", "Cat", "Cats",
        "Cephalopods", "Chicken", "Chickens", "Chimpanzee", "Cows", "Crabs", "Crocodile",
        "Crocodile/Alligator", "Deers", "Dinosaur", "Dog", "Dogs", "Dolphin", "Duo", "Eagles",
        "Elephants", "Fish", "Fishes", "Fox", "Foxes", "Frog", "Frogs", "Giraffe", "Goose",
        "Gorilla", "Hamsters", "Hedgehogs", "Herring", "Hippopotamus", "Horses", "Insect",
        "Insects", "Jellyfish", "Lion", "Llama", "Mammoths", "Meerkats", "Monkey", "Mouse",
        "Octopus", "Orca", "Ostrich", "Owl", "Penguins", "Pigeons", "Pigs", "Rabbit",
        "Random animal family", "Reptiles", "Rhino", "Rodents", "Scorpion", "Sharks", "Sheep",
        "Shrimp", "Skeletons", "Snail", "Snails", "Snake", "Snakes", "Spiders", "Squid",
        "Squirrel", "Starfish", "Tapir", "Tiger", "Tigers", "Turtles", "Whales", "Wolves",
        "Zebras",
    },
    "Food": {
        "Apple", "Avocado", "Banana", "Boba", "Bread", "Broccoli", "Bubble Tea", "Burgers",
        "Cactus", "Candies", "Candy", "Carrot", "Cheese", "Chocolate", "Coconut", "Coffee", "Cola",
        "Cookie", "Cucumber", "Date Fruit", "Desserts", "Dragon Fruit", "Drinks", "Durian",
        "Fanta", "Food", "Food-themed Brainrots", "Fruit", "Fruits", "Gingerbread", "Grape",
        "Grapefruit", "Green Onion", "Grilled Food", "Guava", "Hot Dog", "Hotdog", "Ice Cream",
        "Junk foods", "KFC", "Kebab", "Kiwi", "Lemon", "Mango", "Mangos", "Melon", "Milk", "Onion",
        "Orange", "Pasta", "Pear", "Pepper", "Persimmon", "Pineapple", "Pizza", "Pomegranate",
        "Potato", "Potatoes", "Pumpkin", "Soda", "Spaghetti", "Strawberry", "Sushi", "Sweets",
        "Taco", "Tacos", "Vegetables", "Waffle", "Watermelon", "Zucchini",
    },
    "Object": {
        "Amethyst", "Appliance", "Bathroom", "Bathroom Tools", "Bathtub", "Beaker", "Blenders",
        "Board", "Boots", "Boxes", "Broom", "Canisters", "Cars", "Cellphone", "Chairs", "Clock",
        "Computers", "Cube", "Cubes", "Diamond", "Door", "Electronics", "Fridges", "Furniture",
        "Gems", "Go-Kart", "Gold", "Instruments", "Iron", "Lamp", "Machines",
        "Musical Instruments", "Oven", "Pan", "Papers", "Phone", "Pillow", "Pillows",
        "Refrigerator", "Rocket", "Rockets", "Sapphire", "Shoe", "Shoes", "Showerhead", "Sink",
        "Sombrero", "Television", "Toilets", "Trains", "Truck", "Vehicles", "Washing Machine",
        "Watch", "Wood",
    },
    "Setting": {
        "Arctic", "Beach", "City", "Desert", "Forest", "Garden", "Jungle", "Kitchen", "London",
        "Meadow", "Moon", "Mountain", "Ocean", "Outer Space", "Plains", "Planets", "Plantation",
        "School", "Schools", "Sea", "Sky", "Space", "Space void", "Street", "Sun", "Town", "Tree",
        "Trees",
    },
    "Feline": {
        "Big Cats", "Cat", "Cats", "Lion", "Tiger", "Tigers",
    },
    "Canine": {
        "Dog", "Dogs", "Fox", "Foxes", "Wolves",
    },
    "Primate": {
        "Chimpanzee", "Gorilla", "Monkey",
    },
    "Reptile": {
        "Crocodile", "Crocodile/Alligator", "Reptiles", "Snake", "Snakes", "Turtles",
    },
    "Sea Creature": {
        "Aquatic", "Catfish", "Cephalopods", "Crabs", "Dolphin", "Fish", "Fishes", "Herring",
        "Jellyfish", "Octopus", "Orca", "Shark", "Sharks", "Shrimp", "Squid", "Starfish", "Whales",
    },
    "Bird": {
        "Bird", "Birds", "Chicken", "Chickens", "Eagles", "Goose", "Ostrich", "Owl", "Penguins",
        "Pigeons",
    },
    "Rodent": {
        "Beaver", "Beavers", "Capybara", "Hamsters", "Mouse", "Rodents", "Squirrel",
    },
    "Farm Animal": {
        "Bull", "Cows", "Horses", "Pigs", "Sheep",
    },
    "Big Wild Mammal": {
        "Antelope", "Bears", "Buffalo", "Camels", "Deers", "Elephant", "Elephants", "Giraffe",
        "Hippopotamus", "Llama", "Mammoths", "Rhino", "Tapir", "Zebras",
    },
    "Insect or Bug": {
        "Butterfly", "Insect", "Insects", "Scorpion", "Spiders",
    },
    "Prehistoric": {
        "Dinosaur", "Dinosauro", "Dinosauro Family", "Mammoths", "Prehistoric", "The Dinos",
    },
    "Skeleton or Undead": {
        "Ghost", "Skeleton", "Skeleton Sahur", "Skeletons",
    },
    "Fruit": {
        "Apple", "Avocado", "Banana", "Coconut", "Date Fruit", "Dragon Fruit", "Durian", "Fruit",
        "Fruits", "Grape", "Grapefruit", "Guava", "Kiwi", "Lemon", "Mango", "Mangos", "Melon",
        "Orange", "Pear", "Persimmon", "Pineapple", "Pomegranate", "Strawberry", "Watermelon",
    },
    "Vegetable": {
        "Broccoli", "Cactus", "Carrot", "Cucumber", "Green Onion", "Onion", "Pepper", "Potato",
        "Potatoes", "Pumpkin", "Vegetables", "Zucchini",
    },
    "Dessert or Sweet": {
        "Candies", "Candy", "Chocolate", "Cookie", "Desserts", "Gingerbread", "Ice Cream",
        "Sweets", "Waffle",
    },
    "Drink": {
        "Boba", "Bubble Tea", "Coffee", "Cola", "Drinks", "Fanta", "Milk", "Soda",
    },
    "Fast or Savory Food": {
        "Bread", "Burgers", "Cheese", "Grilled Food", "Hot Dog", "Hotdog", "Junk foods", "KFC",
        "Kebab", "Pasta", "Pizza", "Spaghetti", "Sushi", "Taco", "Tacos",
    },
    "Vehicle": {
        "Bus", "Cars", "Go-Kart", "Rocket", "Rockets", "Trains", "Truck", "Vehicles",
    },
    "Appliance Group": {
        "Appliance", "Blenders", "Fridges", "Machines", "Oven", "Refrigerator", "Washing Machine",
    },
    "Bathroom Item": {
        "Bathroom", "Bathroom Tools", "Bathtub", "Showerhead", "Sink", "Toilets",
    },
    "Electronics": {
        "Cellphone", "Computers", "Electronics", "Phone",
    },
    "Furniture Item": {
        "Chairs", "Furniture", "Pillow", "Pillows",
    },
    "Precious Material": {
        "Amethyst", "Diamond", "Gems", "Gold", "Sapphire",
    },
    "Anthropomorphic": {
        "Anthropomorphic", "Human", "Humanoid Features", "Humans", "Man",
    },
    "Muscular or Strong": {
        "Bodybuilder", "Muscular", "Unstoppable",
    },
    "Loud or Distinct Sound": {
        "High-pitched", "Loud",
    },
    "Dangerous or Scary": {
        "Dangerous", "Explosive", "Horror", "Scary",
    },
    "Immortal or Godlike": {
        "Gods", "Immortal", "Omnipotent",
    },
    "Robotic": {
        "Mutant", "Mutants", "Robots",
    },
    "Combination or Fusion": {
        "Combination", "Combinations", "Fusion", "Hybrids",
    },
    "Group of Multiple Characters": {
        "Couples", "Duo", "Duos", "Trio",
    },
    "Royal": {
        "Kingdom Roles", "Princess", "Royals",
    },
    "Villain": {
        "Assassins", "Bandits", "Betrayer", "Outlaws", "Villains", "Villians",
    },
    "Hero": {
        "Heros", "Warrior",
    },
    "Boss Character": {
        "Boss", "Bosses",
    },
    "Baby or Young Character": {
        "Babies", "Baby", "Bambino",
    },
    "Has Numbers or Letters": {
        "Characters with letters", "Letter", "Letters", "Number", "Numbers",
    },
    "Monster": {
        "Monsters",
    },
    "Fire-Themed": {
        "Fire",
    },
    "Alien or Sci-Fi": {
        "Aliens",
    },
    "Holiday-Themed": {
        "Christmas", "Halloween",
    },
    "Sahur Family": {
        "Alphabet Sahur", "Alphabetsahur", "Mutant Sahur", "Sahur", "Sahur family", "Sahurs",
        "Skeleton Sahur", "Typographic Sahur",
    },
    "Signal Family": {
        "Signal", "Signal Family",
    },
    "Bombardiro or Bomber Family": {
        "Bombardino Family", "Bombardiro Family", "Bomber", "Bombers",
    },
    "Crocodile Family": {
        "Crocaafk", "Crocodildo Family",
    },
    "67 Family": {
        "67 Family",
    },
    "Matteo Family": {
        "Matteo", "Matteo family",
    },
    "Wolf Group": {
        "The Wolf Group members",
    },
}


def _add_supercategories(tags):
    expanded = set(tags)
    for supercategory, members in SUPERCATEGORIES.items():
        if tags & members:
            expanded.add(supercategory)
    return expanded


def _load_characters():
    """Only titles and their (small) tag sets are kept in memory for the process lifetime —
    holding every page's full content in a second permanent copy (on top of what build_index.py
    and chroma_db already need in memory to build/serve the RAG index) contributed to an
    out-of-memory crash on Railway's production instance. Full content is only ever needed for
    a handful of shortlisted candidates at final-guess time, so it's loaded lazily via
    _content_for_titles() instead of held permanently."""
    with open(WIKI_DATA_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    titles = [page["title"] for page in pages]
    tags_by_title = {}
    for page in pages:
        raw_tags = {tag.strip() for tag in CATEGORY_RE.findall(page["content"]) if not _is_excluded_tag(tag.strip())}
        tags_by_title[page["title"]] = _add_supercategories(raw_tags)
    return titles, tags_by_title


ALL_TITLES, TAGS_BY_TITLE = _load_characters()


def _content_for_titles(titles):
    """Re-reads wiki_data.json for just the given titles' content. Called only when making a
    final guess among a small shortlist (at most MAX_SHORTLIST_FOR_GEMINI candidates), never on
    the hot path of narrowing the pool — a full JSON parse is an acceptable one-off cost there,
    much cheaper over the life of the process than holding every page's content in RAM always."""
    wanted = set(titles)
    with open(WIKI_DATA_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    return {page["title"]: page["content"] for page in pages if page["title"] in wanted}


# --- State ---------------------------------------------------------------------------------


class AkinatorState(BaseModel):
    # "revealing": the round ended without a correct guess, and the player is being asked to
    # type in the real character so the round can be broken down against it.
    phase: Literal["asking", "guessing", "revealing"] = "asking"
    # Every title's running score. Absent titles are implicitly 0 (kept sparse only until the
    # first question — after that every title has been nudged one way or the other, so this
    # ends up holding all ~4687 entries in practice; see the module docstring below for why a
    # full dict beats a shrinking hard-filtered list here).
    scores: dict[str, float] = {}
    asked_tags: list[str] = []
    question_count: int = 0
    pending_tag: str | None = None
    pending_guess: str | None = None
    final_guess_made: bool = False
    history: list[str] = []
    # Structured (tag, answer) log of every trait question actually answered this round — kept
    # separate from the free-text `history` above (which is for Gemini prompts) so the reveal
    # breakdown can check each answer against the revealed character's real tags without having
    # to parse history's prose back apart.
    answered_questions: list[dict] = []


# --- Weighted candidate scoring --------------------------------------------------------------
#
# The real Akinator asks with five answers — Yes, Probably, Don't Know, Probably Not, No — not
# a strict three. That isn't cosmetic: it's the difference between a probabilistic model and a
# hard filter. The original version of this module used pure set elimination (a "no" drops
# every candidate with the tag, permanently); that's brittle in exactly the way a real player
# experiences a round — one slightly-too-strict "no" on a trait the player is genuinely unsure
# about permanently kills the correct candidate with no way to recover. Real Akinator instead
# keeps every candidate "alive" the whole round, nudging a running plausibility score up or
# down by each answer, so a wrong or hedged answer costs some ground rather than ending the
# round for that candidate. This is that model: ANSWER_WEIGHTS below define how far each of the
# five answers pushes a candidate's score, matching-candidates one way and non-matching
# candidates the other, symmetrically.

ANSWER_WEIGHTS = {"yes": 2, "probably": 1, "unsure": 0, "probably_not": -1, "no": -2}
TOP_K_FOR_QUESTION_SELECTION = 150
CONFIDENCE_MIN_SCORE = 4
CONFIDENCE_GAP = 4
DISQUALIFIED_SCORE = -1_000_000.0


# --- Learned tag reliability (crowd-sourced from revealed rounds) --------------------------
#
# TAG_STATS accumulates, across every round any player has ever revealed, how often each tag's
# answer actually held up against the real character (see _record_reveal_outcomes, called from
# _analyze_reveal below). _tag_reliability turns that into a multiplier on the tag's normal
# ANSWER_WEIGHTS delta — this is what lets the game's accuracy improve over time from real play
# instead of staying fixed at whatever the static wiki-derived tags alone provide.

_learning_lock = threading.Lock()


def _load_tag_stats():
    try:
        with open(LEARNING_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


TAG_STATS = _load_tag_stats()


def _save_tag_stats():
    # Atomic write (temp file + rename) so a crash or a concurrent request reading this file
    # mid-write can never see a half-written, corrupt learning file.
    tmp_path = LEARNING_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(TAG_STATS, f)
    os.replace(tmp_path, LEARNING_FILE)


def _tag_reliability(tag):
    """Laplace-smoothed reliability multiplier in (0.5, 1.5) — exactly 1.0 with no data yet
    (falls back to the tag's plain ANSWER_WEIGHTS delta), climbing toward 1.5 for a tag that
    keeps proving consistent across many revealed rounds, and down toward 0.5 for one that keeps
    proving unreliable. The +1/+2 smoothing keeps a single early round from swinging a tag to an
    extreme before enough data has actually accumulated."""
    stats = TAG_STATS.get(tag)
    if not stats:
        return 1.0
    total = stats["consistent"] + stats["inconsistent"]
    ratio = (stats["consistent"] + 1) / (total + 2)
    return 0.5 + ratio


def _record_reveal_outcomes(consistency_by_tag):
    """Persists one revealed round's outcomes. consistency_by_tag maps tag -> list of bools
    (True = the player's answer matched the revealed character's real tag); only answers that
    made a claim (weight != 0) are included — "unsure" carries no signal either way."""
    if not consistency_by_tag:
        return
    with _learning_lock:
        for tag, results in consistency_by_tag.items():
            stats = TAG_STATS.setdefault(tag, {"consistent": 0, "inconsistent": 0})
            for consistent in results:
                stats["consistent" if consistent else "inconsistent"] += 1
        _save_tag_stats()


def _apply_score_update(scores, tag, answer):
    weight = ANSWER_WEIGHTS.get(answer, 0)
    if weight == 0 or tag is None:
        return dict(scores)
    scaled_weight = weight * _tag_reliability(tag)
    updated = dict(scores)
    for title in ALL_TITLES:
        has_tag = tag in TAGS_BY_TITLE.get(title, ())
        delta = scaled_weight if has_tag else -scaled_weight
        updated[title] = updated.get(title, 0) + delta
    return updated


def _ranked_titles(scores):
    """All real titles, highest score first; ties broken by wiki order for determinism."""
    indexed = sorted(range(len(ALL_TITLES)), key=lambda i: (-scores.get(ALL_TITLES[i], 0), i))
    return [ALL_TITLES[i] for i in indexed]


def _confident_enough(ranked_titles, scores):
    if len(ranked_titles) < 2:
        return True
    top_score = scores.get(ranked_titles[0], 0)
    second_score = scores.get(ranked_titles[1], 0)
    return top_score >= CONFIDENCE_MIN_SCORE and (top_score - second_score) >= CONFIDENCE_GAP


# The 4 original, broadest SUPERCATEGORIES entries — asked ahead of every narrower category
# (Feline/Bird/Fruit/...) or raw tag by _best_split_tag below, mirroring how a player naturally
# narrows a guess themselves: coarse first ("is it an animal?"), specific once the coarse answer
# is known ("okay, what kind?"). Narrower tags often split the CURRENT candidates more evenly on
# paper, but a broad question asked first is more legible round to round and avoids a lucky-split
# raw tag or niche subcategory jumping the queue ahead of the categories a player expects first.
BROAD_CATEGORIES = {"Animal", "Food", "Object", "Setting"}


def _best_split_tag(candidate_titles, asked_tags):
    """The tag whose presence among the current leaderboard is closest to a 50/50 split carries
    the most information about which half the answer will fall into — the same greedy heuristic
    a real 20-questions solver uses, now applied to the current top-scoring candidates rather
    than a shrinking hard-filtered pool, so question selection stays focused on what actually
    distinguishes the leaders. Already-asked tags are excluded so we never repeat a question.
    Returns None once no remaining tag usefully splits the set (e.g. every candidate considered
    happens to share the same tags, or none are tagged at all).

    Tries BROAD_CATEGORIES first and only falls back to the full tag pool (narrower
    supercategories and raw wiki tags alike) once every broad category has either already been
    asked or no longer usefully splits the current candidates — see BROAD_CATEGORIES above."""
    asked = set(asked_tags)
    tag_counts = {}
    for title in candidate_titles:
        for tag in TAGS_BY_TITLE.get(title, ()):
            if tag in asked:
                continue
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    half = len(candidate_titles) / 2

    def best_of(pool):
        useful = {tag: count for tag, count in pool.items() if 0 < count < len(candidate_titles)}
        if not useful:
            return None
        return min(useful, key=lambda tag: abs(useful[tag] - half))

    broad_pool = {tag: count for tag, count in tag_counts.items() if tag in BROAD_CATEGORIES}
    broad_pick = best_of(broad_pool)
    return broad_pick if broad_pick is not None else best_of(tag_counts)


# --- Gemini calls (with deterministic fallbacks if they fail) ------------------------------


class QuestionPhrasing(BaseModel):
    question_text: str


QUESTION_SYSTEM_INSTRUCTION = (
    "You are the host of a 20-questions-style guessing game about Italian Brainrot wiki "
    "characters. You are given ONE trait (a wiki category tag) that the game engine has "
    "already chosen as the most useful thing to ask about next — your only job is to phrase "
    "it as one short, natural yes/no question a player can answer at a glance. Never mention "
    "'category' or 'tag' — phrase it as a normal trait question about appearance, species/base "
    "object, or a similar observable trait. Keep it under 20 words. Never introduce a trait "
    "other than the one given, and never reference or guess a specific character name."
)


def _fallback_question_text(tag):
    return f"Does your character have anything to do with '{tag}' (e.g. {tag.lower()}-themed or {tag.lower()}-related)?"


def _phrase_question(gemini_client, tag, history_lines):
    history_summary = "\n".join(history_lines[-10:]) or "(none yet)"
    prompt = (
        f"Trait to ask about: {tag}\n\n"
        f"Questions already asked this round (for phrasing variety, don't repeat wording):\n"
        f"{history_summary}\n\n"
        f"Write one short yes/no question asking whether the player's character has this trait."
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=QUESTION_SYSTEM_INSTRUCTION,
                max_output_tokens=QUESTION_MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=QUESTION_THINKING_BUDGET),
                response_mime_type="application/json",
                response_schema=QuestionPhrasing,
            ),
        )
    except genai_errors.APIError:
        return _fallback_question_text(tag)

    parsed = response.parsed
    if parsed is None or not parsed.question_text.strip():
        return _fallback_question_text(tag)
    return parsed.question_text.strip()


class GuessPick(BaseModel):
    guess_title: str


GUESS_SYSTEM_INSTRUCTION = (
    "You are the host of a 20-questions-style guessing game about Italian Brainrot wiki "
    "characters, making a final guess from a short list of real candidates. You must pick "
    "EXACTLY one title, copied verbatim from the candidate list given — never invent or "
    "modify a name, and never pick anything not in that list."
)


def _pick_best_guess(gemini_client, shortlist_titles, history_lines):
    history_summary = "\n".join(history_lines[-10:]) or "(no questions asked yet)"
    content_by_title = _content_for_titles(shortlist_titles)
    context = "\n\n---\n\n".join(
        f"[{title}]\n{content_by_title.get(title, '')[:CONTENT_TRUNCATE_CHARS]}" for title in shortlist_titles
    )
    prompt = (
        f"Candidates (choose ONLY one of these, copied exactly):\n"
        + "\n".join(f"- {title}" for title in shortlist_titles)
        + f"\n\nWiki context on each candidate:\n{context}\n\n"
        f"Questions asked this round and the player's answers:\n{history_summary}\n\n"
        f"Pick the single candidate that best matches all the answers given."
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GUESS_SYSTEM_INSTRUCTION,
                max_output_tokens=GUESS_MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=GUESS_THINKING_BUDGET),
                response_mime_type="application/json",
                response_schema=GuessPick,
            ),
        )
    except genai_errors.APIError:
        return shortlist_titles[0]

    parsed = response.parsed
    if parsed is None:
        return shortlist_titles[0]

    lookup = {title.strip().lower(): title for title in shortlist_titles}
    return lookup.get(parsed.guess_title.strip().lower(), shortlist_titles[0])


# --- Reveal & post-round analysis -----------------------------------------------------------


ANSWER_DISPLAY = {"yes": "Yes", "probably": "Probably", "unsure": "Don't Know", "probably_not": "Probably Not", "no": "No"}
REVEAL_PROMPT = " Type the character's real name below and I'll show you where I went wrong (or start a new round)."


def _find_title(revealed_name):
    """Matches the player's typed reveal against a real wiki title — case-insensitive exact
    match first, falling back to a substring match only when it's unambiguous. Never invents or
    guesses a title that doesn't exist in the wiki."""
    name = (revealed_name or "").strip()
    if not name:
        return None
    exact = {title.lower(): title for title in ALL_TITLES}.get(name.lower())
    if exact:
        return exact
    substring_matches = [title for title in ALL_TITLES if name.lower() in title.lower()]
    return substring_matches[0] if len(substring_matches) == 1 else None


def _analyze_reveal(revealed_title, answered_questions):
    """Checks each answered trait question against the revealed character's actual tags and
    reports which ones look like they were answered inconsistently — the concrete "where did I
    go wrong" breakdown the player asked for. As a side effect, persists each claim-making
    answer's consistency to the learned tag-reliability store (see _record_reveal_outcomes) so
    future rounds' scoring benefits from this round's outcome too."""
    real_tags = TAGS_BY_TITLE.get(revealed_title, set())
    lines = []
    wrong_count = 0
    consistency_by_tag = {}
    for entry in answered_questions:
        tag, answer = entry.get("tag"), entry.get("answer")
        weight = ANSWER_WEIGHTS.get(answer, 0)
        has_tag = tag in real_tags
        answer_label = ANSWER_DISPLAY.get(answer, answer)
        if weight == 0:
            lines.append(f"- {tag}? You said {answer_label} — no claim made, doesn't affect scoring.")
            continue
        consistent = (weight > 0) == has_tag
        consistency_by_tag.setdefault(tag, []).append(consistent)
        trait_note = "has" if has_tag else "doesn't have"
        if consistent:
            lines.append(f"- {tag}? You said {answer_label} — ✅ consistent ({revealed_title} {trait_note} this trait).")
        else:
            wrong_count += 1
            lines.append(f"- {tag}? You said {answer_label} — ❌ likely wrong ({revealed_title} {trait_note} this trait).")

    _record_reveal_outcomes(consistency_by_tag)

    header = f"It was **{revealed_title}**! Here's how your answers lined up:\n\n"
    if not answered_questions:
        footer = "\n\n(No trait questions were answered this round.)"
    elif wrong_count == 0:
        footer = "\n\nAll your answers were consistent with the real character — I just didn't land on a confident enough guess in time."
    else:
        if wrong_count == 1:
            footer = "\n\n1 answer looks like it may have thrown me off."
        else:
            footer = f"\n\n{wrong_count} answers look like they may have thrown me off."
    return header + "\n".join(lines) + footer


def _handle_reveal(state, revealed_name):
    title = _find_title(revealed_name)
    if title is None:
        return (
            "I couldn't find that character in the wiki, so I can't break down the round — but "
            "thanks for telling me! Start a new round whenever you're ready."
        ), None
    return _analyze_reveal(title, state.answered_questions), None


# --- Turn orchestration ----------------------------------------------------------------------


def _make_guess(gemini_client, scores, ranked_titles, asked_tags, question_count, history, final_guess_made, answered_questions):
    shortlist = ranked_titles[:MAX_SHORTLIST_FOR_GEMINI]
    guess = shortlist[0] if len(shortlist) == 1 else _pick_best_guess(gemini_client, shortlist, history)
    new_state = AkinatorState(
        phase="guessing",
        scores=scores,
        asked_tags=asked_tags,
        question_count=question_count,
        pending_guess=guess,
        final_guess_made=final_guess_made,
        history=history,
        answered_questions=answered_questions,
    )
    return f"Is it **{guess}**?", new_state


def _advance(gemini_client, scores, asked_tags, question_count, history, final_guess_made, answered_questions):
    ranked_titles = _ranked_titles(scores)
    top_score = scores.get(ranked_titles[0], 0)

    # Every real candidate has been pushed at or below the disqualification floor — every
    # wrong guess demotes one this far, so this only happens once we've truly run out of
    # plausible candidates, not just a low-information start.
    if top_score <= DISQUALIFIED_SCORE:
        message = "None of the real wiki characters fit all those answers — I'm stuck!" + REVEAL_PROMPT
        new_state = AkinatorState(
            phase="revealing", asked_tags=asked_tags, question_count=question_count,
            final_guess_made=final_guess_made, history=history, answered_questions=answered_questions,
        )
        return message, new_state

    if question_count >= MAX_QUESTIONS:
        return _make_guess(
            gemini_client, scores, ranked_titles, asked_tags, question_count, history,
            final_guess_made=True, answered_questions=answered_questions,
        )

    if _confident_enough(ranked_titles, scores):
        return _make_guess(
            gemini_client, scores, ranked_titles, asked_tags, question_count, history,
            final_guess_made, answered_questions,
        )

    # Question selection focuses on the current leaders, not the full population — after the
    # opening question (all scores still tied at 0) there's no meaningful "leaderboard" yet, so
    # that first pick alone still draws from everyone.
    active = ALL_TITLES if question_count == 0 else ranked_titles[:TOP_K_FOR_QUESTION_SELECTION]
    tag = _best_split_tag(active, asked_tags)
    if tag is None:
        return _make_guess(
            gemini_client, scores, ranked_titles, asked_tags, question_count, history,
            final_guess_made, answered_questions,
        )

    question_text = _phrase_question(gemini_client, tag, history)
    new_state = AkinatorState(
        phase="asking",
        scores=scores,
        asked_tags=asked_tags + [tag],
        question_count=question_count,
        pending_tag=tag,
        final_guess_made=final_guess_made,
        history=history,
        answered_questions=answered_questions,
    )
    return question_text, new_state


def _handle_question_answer(gemini_client, state, answer):
    scores = _apply_score_update(state.scores, state.pending_tag, answer)
    question_count = state.question_count + 1
    history = state.history + [f"Q: {state.pending_tag}? A: {answer}"]
    answered_questions = state.answered_questions + [{"tag": state.pending_tag, "answer": answer}]
    return _advance(gemini_client, scores, state.asked_tags, question_count, history, state.final_guess_made, answered_questions)


def _handle_guess_answer(gemini_client, state, answer):
    history = state.history + [f"Guess: {state.pending_guess}? A: {answer}"]
    if answer == "yes":
        return f"Got it — it was **{state.pending_guess}**! 🎉", None

    if state.final_guess_made:
        message = f"You've stumped me! {MAX_QUESTIONS} questions and my final guess weren't enough." + REVEAL_PROMPT
        new_state = AkinatorState(
            phase="revealing", asked_tags=state.asked_tags, question_count=state.question_count,
            final_guess_made=state.final_guess_made, history=history, answered_questions=state.answered_questions,
        )
        return message, new_state

    # A wrong guess is strong evidence against that specific candidate — disqualify it outright
    # (rather than just nudging its score down) so it's never guessed again this round, without
    # otherwise disturbing every other candidate's accumulated evidence.
    scores = dict(state.scores)
    scores[state.pending_guess] = DISQUALIFIED_SCORE
    return _advance(
        gemini_client, scores, state.asked_tags, state.question_count, history,
        state.final_guess_made, state.answered_questions,
    )


def process_turn(gemini_client, state_dict, answer, revealed_name=None):
    """Entry point. state_dict is the previous turn's returned state (None to start a new
    round), answer is "yes" | "probably" | "unsure" | "probably_not" | "no" | "reset" from the
    player's last button press. revealed_name is only used in the "revealing" phase — the real
    character name the player typed in after a round ended without a correct guess. Returns
    (message_text, new_state_dict_or_None) — None means the round has ended."""
    answer = (answer or "").strip().lower()

    if answer == "reset":
        return "Round ended — start a new one whenever you're ready!", None

    if state_dict is not None:
        try:
            state = AkinatorState(**state_dict)
        except (TypeError, ValueError):
            state = None
    else:
        state = None

    if state is None:
        message, new_state = _advance(gemini_client, {}, [], 0, [], False, [])
        return message, (new_state.model_dump() if new_state else None)

    if state.phase == "revealing":
        message, new_state = _handle_reveal(state, revealed_name)
        return message, (new_state.model_dump() if new_state else None)

    if state.phase == "asking" and answer not in ANSWER_WEIGHTS:
        answer = "unsure"

    if state.phase == "asking":
        message, new_state = _handle_question_answer(gemini_client, state, answer)
    else:
        # A guess is a binary claim — only an explicit "yes" confirms it; every other answer
        # (no, unsure, probably*) rejects it and moves on, so the 5-point scale doesn't need
        # separate handling here.
        message, new_state = _handle_guess_answer(gemini_client, state, answer)

    return message, (new_state.model_dump() if new_state else None)
