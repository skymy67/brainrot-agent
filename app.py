#!/usr/bin/env python3
"""FastAPI backend that answers questions via RAG over the Italian Brainrot wiki."""

import base64
import json
import os
import re
import urllib.error
import urllib.request

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import akinator_mode
import craft_mode
import evolution_mode
import quest_mode
import rarity_mode
import rpg_mode
from content_policy import with_content_policy

# Points at a Railway volume in production (set via the DATA_DIR env var) so chroma_db survives
# redeploys instead of rebuilding from scratch on every one; defaults to the working directory
# for local dev, where it already lived. akinator_mode.py reads the same env var for its own
# persistent learning store, so both end up on the same volume.
CHROMA_DIR = os.path.join(os.environ.get("DATA_DIR", "."), "chroma_db")
COLLECTION_NAME = "wiki_pages"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-3.6-flash"
TOP_K = 5
# BGE models recommend this instruction prefix on queries (not on indexed documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODES = {
    "qa": {
        "system_instruction": with_content_policy(
            "You are a helpful assistant answering questions about the Italian Brainrot wiki "
            "using the provided context."
        ),
        "instruction": (
            "Answer the question using the context above. A character's name in the context may "
            "not exactly match how the user spelled it — typos, a doubled/missing letter, or "
            "different capitalization are common. If a context character is clearly the same one "
            "the user means despite a small spelling difference, treat it as a match and answer "
            "normally using that character's context; don't reject it just because the spelling "
            "isn't byte-for-byte identical. Only say the context doesn't contain the answer when "
            "there's genuinely no matching or relevant character in it."
        ),
        "thinking_budget": 256,
        "max_output_tokens": 1536,
    },
    "creative": {
        "system_instruction": with_content_policy(
            "You are a creative storyteller for the Italian Brainrot universe. Treat the provided "
            "wiki context as canon — character traits, lore, and relationships should stay consistent "
            "with it — but you're free to invent scenes, dialogue, and details the wiki doesn't cover "
            "in order to tell an engaging story."
        ),
        "instruction": (
            "Use the canon details in the context above to stay accurate to the wiki's lore, "
            "characters, and relationships. Then write a creative, engaging response to the request "
            "below — you may invent scenes and details not in the wiki as long as they don't "
            "contradict the canon facts provided."
        ),
        "thinking_budget": 512,
        "max_output_tokens": 3072,
    },
}

# Matches a question asking what a character looks like, so the visual-description feature only
# spends the extra fetch/multimodal-call cost when it's actually likely to help — most questions
# aren't about appearance, and the text-only RAG context already answers those fine on its own.
APPEARANCE_INTENT_RE = re.compile(
    r"(?i)\b(look|looks|looking|appearance|appear|picture|photo|image|visual(ly)?|drawn|drawing)\b"
)
IMAGE_FETCH_TIMEOUT = 8
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}


def _load_image_urls():
    # A separate, lightweight (title -> URL string) read of wiki_data.json — akinator_mode.py
    # already does its own independent read of the same file for its own purposes (titles/tags),
    # so this follows the same established pattern rather than threading a shared load through
    # both modules. Only string URLs are kept in memory, not page content, so this doesn't
    # reintroduce the full-content memory duplication the OOM fix earlier removed.
    with open("wiki_data.json", encoding="utf-8") as f:
        pages = json.load(f)
    return {page["title"]: page["image_url"] for page in pages if "image_url" in page}


IMAGE_URL_BY_TITLE = _load_image_urls()


def best_title_match(question, metadatas):
    """The RAG top match isn't reliably the exact character asked about — measured directly:
    querying this wiki's own embeddings for the literal string "Tralalero Tralala" returns
    "Brololino Bralila" as the #1 result (a differently-named, differently-gibberish character),
    with the real Tralalero Tralala only showing up at rank 4. Trusting metadatas[0] blindly for
    an image lookup would show Gemini a real image while claiming — wrongly — that it's the
    portrait of the character the player actually asked about. Two tiers before falling back to
    the top RAG rank: an exact (case-insensitive) title match — the common case for modes where
    the question IS just a bare character name (Craft/RPG/Rarity/Evolution) — and, for a full
    natural-language question a bare-name match can't catch ("what does X look like?"), a real
    title appearing verbatim inside the question, only when exactly one candidate qualifies (the
    same exact-then-unambiguous-substring pattern akinator_mode._find_title already uses)."""
    if not metadatas:
        return None
    question_lower = question.strip().lower()
    for meta in metadatas:
        if meta["title"].strip().lower() == question_lower:
            return meta["title"]
    substring_matches = [meta["title"] for meta in metadatas if meta["title"].strip().lower() in question_lower]
    if len(substring_matches) == 1:
        return substring_matches[0]
    return metadatas[0]["title"]


def fetch_character_image(image_url, timeout=IMAGE_FETCH_TIMEOUT):
    """Best-effort fetch of a character's real wiki portrait for a visual-description question.
    Live wiki access is already known to be blocked from some environments (this sandbox
    included — see the earlier wiki-scraping and Akinator research this session), and Railway's
    own network access to it hasn't been verified, so this must never let a fetch failure
    surface as an error: any problem (network error, timeout, non-200 status, an extension with
    no known MIME type) returns (None, None) so the caller falls back to a text-only
    description instead of crashing or hanging the request."""
    mime_type = IMAGE_MIME_TYPES.get(os.path.splitext(image_url)[1].lower())
    if mime_type is None:
        print(f"[image fetch] skipped, unrecognized extension: {image_url}")
        return None, None
    try:
        request = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                print(f"[image fetch] non-200 status {response.status}: {image_url}")
                return None, None
            data = response.read()
            print(f"[image fetch] succeeded, {len(data)} bytes: {image_url}")
            return data, mime_type
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"[image fetch] failed ({exc}): {image_url}")
        return None, None


app = FastAPI(title="Italian Brainrot Wiki Chat")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def _load_or_build_collection():
    # chroma_db/ is a gitignored build artifact — rebuild it from wiki_data.json on first boot
    # (e.g. a fresh Railway deploy) instead of requiring it to be committed to the repo. Rebuild
    # not just when the collection is missing outright, but also when it exists with zero
    # documents — e.g. a persisted volume left over from a deploy whose build got interrupted
    # partway through, which get_collection() alone can't detect since it doesn't raise.
    #
    # Each chromadb.PersistentClient() call here is intentionally a short-lived temporary,
    # never held in a variable across the build_index.main() call below — build_index.main()
    # opens its own PersistentClient on the same on-disk path, and keeping an older client
    # instance alive at the same time previously caused a startup crash on the SQLite-backed
    # store when the rebuild path actually ran.
    try:
        existing = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        existing = None

    if existing is not None and existing.count() > 0:
        return existing

    import build_index

    # Reuses embedding_model (already loaded above) instead of letting build_index.py load its
    # own separate copy of the same SentenceTransformer — that duplication contributed to an
    # out-of-memory crash in production.
    build_index.main(model=embedding_model)
    return chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)


collection = _load_or_build_collection()

gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment


class ChatRequest(BaseModel):
    question: str
    mode: str = "qa"
    image_base64: str | None = None
    image_mime_type: str | None = None
    # Akinator Mode only: "yes" | "no" | "unsure" | "reset" from the player's last button press,
    # and the state this module last returned (None to start a new round). Everything else the
    # backend does is a single stateless request/response, so — like the frontend already does
    # for chat history via sessionStorage — round state round-trips through the client instead
    # of living in server memory.
    akinator_answer: str | None = None
    akinator_state: dict | None = None
    # Quest Mode only: same round-tripped-state pattern as Akinator above. question doubles as
    # the adventure brief on the first turn and the player's next move/steer on later turns.
    quest_state: dict | None = None


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    akinator_state: dict | None = None
    quest_state: dict | None = None


def retrieve_chunks(question, top_k=TOP_K):
    query_embedding = embedding_model.encode([QUERY_PREFIX + question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


def retrieve_evolution_chunks(character_name, top_k=8):
    """Plain-name retrieval alone rarely surfaces a real baby/evolved variant — e.g. a query for
    "Bombardiro Crocodilo" doesn't return "Los Crocodilitos Dicen Kaboom" in the top 15 results,
    even though that page explicitly describes itself as a baby version of it. Rephrasing the
    query toward the relationship we're looking for (tested directly) reliably surfaces it. Runs
    three targeted queries and merges them, deduped by title, to build the candidate pool that
    Evolution Mode is restricted to choosing real prevolution/evolution stages from — a wider
    top_k than other modes use, since a real match that isn't retrieved here can never be found
    (the model is never allowed to invent one instead)."""
    queries = [
        character_name,
        f"baby form prevolution of {character_name}",
        f"evolved mature final form of {character_name}",
    ]
    documents, metadatas, seen_titles = [], [], set()
    for query in queries:
        docs, metas = retrieve_chunks(query, top_k=top_k)
        for doc, meta in zip(docs, metas):
            if meta["title"] not in seen_titles:
                seen_titles.add(meta["title"])
                documents.append(doc)
                metadatas.append(meta)
    return documents, metadatas


def evolution_candidate_titles(character_name, metadatas):
    seen, candidates = set(), []
    target = character_name.strip().lower()
    for meta in metadatas:
        title = meta["title"]
        if title.strip().lower() != target and title not in seen:
            seen.add(title)
            candidates.append(title)
    return candidates


def dedupe_sources(metadatas):
    seen = set()
    sources = []
    for meta in metadatas:
        key = (meta["title"], meta["url"])
        if key not in seen:
            seen.add(key)
            sources.append(Source(title=meta["title"], url=meta["url"]))
    return sources


def handle_gemini_errors(fn):
    try:
        return fn()
    except genai_errors.APIError as exc:
        if exc.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini API rate limit reached. Wait a bit and try again.",
            ) from exc
        if exc.code == 503:
            raise HTTPException(
                status_code=503,
                detail="Gemini is temporarily overloaded. Try again in a moment.",
            ) from exc
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.mode == "rarity":
        documents, metadatas = retrieve_chunks(request.question)
        context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))

        image_bytes = None
        if request.image_base64:
            try:
                image_bytes = base64.b64decode(request.image_base64)
            except (base64.binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail="Invalid image data.") from exc

        answer = handle_gemini_errors(
            lambda: rarity_mode.run_rarity_pipeline(
                gemini_client,
                request.question,
                context,
                image_bytes=image_bytes,
                image_mime_type=request.image_mime_type,
            )
        )
        return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))

    if request.mode == "evolution":
        documents, metadatas = retrieve_evolution_chunks(request.question)
        context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))
        candidate_titles = evolution_candidate_titles(request.question, metadatas)

        answer = handle_gemini_errors(
            lambda: evolution_mode.build_evolution_line(gemini_client, request.question, context, candidate_titles)
        )
        return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))

    if request.mode == "rpg":
        documents, metadatas = retrieve_chunks(request.question)
        context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))

        answer = handle_gemini_errors(
            lambda: rpg_mode.build_dex_entry(gemini_client, request.question, context)
        )
        return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))

    if request.mode == "craft":
        documents, metadatas = retrieve_chunks(request.question)
        context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))

        # Unlike the visual-description feature in the generic mode path below, the image fetch
        # here is never gated behind an appearance-intent check — confirming components visually
        # is this mode's whole purpose, not an occasional add-on, so it's always attempted when
        # best_title_match() resolves to a known image_url. fetch_character_image() already
        # degrades to (None, None) on any failure, so a blocked/missing image just falls back to
        # text-only reasoning rather than failing the request.
        image_bytes, image_mime_type = None, None
        if metadatas:
            image_url = IMAGE_URL_BY_TITLE.get(best_title_match(request.question, metadatas))
            if image_url:
                image_bytes, image_mime_type = fetch_character_image(image_url)

        answer = handle_gemini_errors(
            lambda: craft_mode.build_recipe(gemini_client, request.question, context, image_bytes, image_mime_type)
        )
        return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))

    if request.mode == "akinator":
        # No RAG retrieval here — candidate narrowing works over the full wiki character list
        # akinator_mode.py loads itself, not a per-question chroma_db lookup, so there are no
        # "sources" to report for this mode. request.question doubles as the reveal-phase free
        # text (the real character name the player types after a round ends without a win) —
        # it's otherwise unused by this mode, since every other turn is a button press.
        answer, akinator_state = handle_gemini_errors(
            lambda: akinator_mode.process_turn(
                gemini_client, request.akinator_state, request.akinator_answer, request.question
            )
        )
        return ChatResponse(answer=answer, sources=[], akinator_state=akinator_state)

    if request.mode == "quest":
        # Stateful and round-tripped through the client exactly like Akinator Mode above.
        # request.question doubles as the adventure brief (no quest_state yet) or the player's
        # next move/steer (mid-campaign). Retrieval stays anchored to the campaign's own goal on
        # later turns (rather than just the latest short player message) so it keeps surfacing
        # thematically relevant — but still unseen — Brainrots chapter after chapter.
        prior_state = request.quest_state
        query = request.question if prior_state is None else f"{prior_state.get('goal', '')} {request.question}".strip()
        documents, metadatas = retrieve_chunks(query, top_k=quest_mode.CANDIDATES_PER_CHAPTER)

        answer, quest_state = handle_gemini_errors(
            lambda: quest_mode.process_turn(
                gemini_client, prior_state, request.question, documents, metadatas
            )
        )
        return ChatResponse(answer=answer, sources=dedupe_sources(metadatas), quest_state=quest_state)

    if request.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{request.mode}'.")
    mode = MODES[request.mode]

    documents, metadatas = retrieve_chunks(request.question)
    context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))

    user_message = f"Context from the Italian Brainrot wiki:\n\n{context}\n\nRequest: {request.question}\n\n{mode['instruction']}"

    # Visual-description feature: for an appearance question ("what does X look like?"), try
    # fetching the top-matched character's real wiki portrait and hand it to Gemini alongside
    # the text context, so the description is grounded in the actual image rather than guessed
    # from prose alone. fetch_character_image() already degrades to (None, None) on any failure
    # (blocked network, timeout, missing image), so this only changes the request shape when a
    # real image was actually retrieved — otherwise it's the exact same text-only call as before.
    image_bytes, image_mime_type, image_title = None, None, None
    if metadatas and APPEARANCE_INTENT_RE.search(request.question):
        image_title = best_title_match(request.question, metadatas)
        image_url = IMAGE_URL_BY_TITLE.get(image_title)
        if image_url:
            image_bytes, image_mime_type = fetch_character_image(image_url)

    if image_bytes:
        contents = types.Content(
            parts=[
                types.Part.from_text(
                    text=(
                        f"The image below is {image_title}'s actual wiki portrait — describe "
                        "what it actually shows (colors, shape, pose, expression, setting) "
                        "rather than relying only on the text context for appearance details."
                    )
                ),
                types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type),
                types.Part.from_text(text=user_message),
            ]
        )
    else:
        contents = user_message

    response = handle_gemini_errors(
        lambda: gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=mode["system_instruction"],
                max_output_tokens=mode["max_output_tokens"],
                thinking_config=types.ThinkingConfig(thinking_budget=mode["thinking_budget"]),
            ),
        )
    )

    answer = response.text or "The model didn't return a response — try asking again."

    return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))


# Serves static/index.html at "/" — registered after /chat so the API route takes priority.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
