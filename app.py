#!/usr/bin/env python3
"""FastAPI backend that answers questions via RAG over the Italian Brainrot wiki."""

import base64

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import rarity_mode

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "wiki_pages"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-3.6-flash"
TOP_K = 5
# BGE models recommend this instruction prefix on queries (not on indexed documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODES = {
    "qa": {
        "system_instruction": (
            "You are a helpful assistant answering questions about the Italian Brainrot wiki "
            "using the provided context."
        ),
        "instruction": (
            "Answer the question using only the context above. "
            "If the context doesn't contain the answer, say so."
        ),
        "thinking_budget": 256,
        "max_output_tokens": 1536,
    },
    "creative": {
        "system_instruction": (
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
    "rpg": {
        "system_instruction": (
            "You are an RPG stat generator for the Italian Brainrot universe. Given wiki context "
            "about a character, you produce a balanced RPG character sheet: HP, Speed, Attack, "
            "Special Attack, and Defense, each on a 1-100 scale, plus a Special — a named signature "
            "move or ability with a one-line effect description. Base every stat on documented "
            "abilities, combat feats, size, and power scaling from the context. If the context "
            "doesn't describe abilities or a power level directly, infer reasonable stats from the "
            "character's physical appearance, size, and apparent strength as described. Keep stats "
            "consistent with comparable characters — don't inflate everything to the max."
        ),
        "instruction": (
            "Using the canon details above, generate an RPG character sheet for this character in "
            "this exact format:\n\n"
            "**[Character Name] — RPG Stats**\n"
            "- HP: X/100\n"
            "- Speed: X/100\n"
            "- Attack: X/100\n"
            "- Special Attack: X/100\n"
            "- Defense: X/100\n"
            "- Special: [Move Name] — [one-line effect]\n\n"
            "**Reasoning:** briefly justify each stat, tying it to specific lore, abilities, or "
            "appearance details from the context. If a stat had to be inferred from appearance "
            "rather than documented lore, say so."
        ),
        "thinking_budget": 512,
        "max_output_tokens": 2560,
    },
    "evolution": {
        "system_instruction": (
            "You are an evolution-line designer for the Italian Brainrot universe, in the style "
            "of a Pokémon-style evolution chain. Given wiki context about a character, you "
            "determine where it logically sits in an evolution line and invent the missing "
            "stage(s), using consistent Italian-brainrot-style naming — nonsense Italian-sounding "
            "compound names that stay thematically consistent with the character's animal/object/"
            "concept base. A line has 2 or 3 total stages — decide based on what's thematically "
            "logical for this specific character, never force a fixed count. If no logically "
            "consistent prevolution or evolution exists, say so directly instead of fabricating "
            "a forced chain."
        ),
        "instruction": (
            "Using the character's name and the wiki context above, determine where this brainrot "
            "would logically sit in an evolution line:\n"
            "- Simple/diminutive/'baby'-style name or minimal lore presence → treat it as Stage 1 "
            "and invent the evolution(s) that follow.\n"
            "- Elaborate name, strong lore presence, or a high power tier → treat it as the FINAL "
            "stage and invent the earlier prevolution stage(s) that would logically precede it.\n"
            "- Plausible middle stage → invent both a prevolution and an evolution.\n\n"
            "Construct a line of 2 or 3 total stages. Each invented stage's name must match the "
            "wiki's Italian-brainrot naming convention and stay thematically consistent with this "
            "character's animal/object/concept base.\n\n"
            "Assign a level requirement to each stage: stage 1 is always Lv 1. Later stages "
            "require meaningfully higher levels, roughly: stage 2 ~ Lv 10-16, stage 3 ~ Lv 25-36. "
            "Nudge these slightly higher if the context indicates a high rarity/power tier (e.g. "
            "Legendary, Mythic, Brainrot God, Secret) — keep it simple, don't overthink it.\n\n"
            "If you cannot construct a logically consistent evolution line for this character, "
            "respond with ONLY one line: '[Character Name] has no evolution line.' Do not "
            "fabricate a forced chain.\n\n"
            "Otherwise, output ONLY the stages in order from earliest to final, in this exact "
            "format, with no descriptions, lore, or extra text:\n\n"
            "Stage 1: [Name] — Lv 1\n"
            "Stage 2: [Name] — Lv [X]\n"
            "Stage 3: [Name] — Lv [X]  (omit this line entirely if the line only has 2 stages)"
        ),
        "thinking_budget": 384,
        "max_output_tokens": 768,
    },
}

app = FastAPI(title="Italian Brainrot Wiki Chat")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment


class ChatRequest(BaseModel):
    question: str
    mode: str = "qa"
    image_base64: str | None = None
    image_mime_type: str | None = None


class Source(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def retrieve_chunks(question, top_k=TOP_K):
    query_embedding = embedding_model.encode([QUERY_PREFIX + question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


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

    if request.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{request.mode}'.")
    mode = MODES[request.mode]

    documents, metadatas = retrieve_chunks(request.question)

    context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))

    if request.mode == "evolution":
        # Free, no extra API call — reuses the same tier lookup Rarity Mode already built,
        # to nudge the level curve per the spec's "scale slightly based on rarity/tier" rule.
        known_tier = rarity_mode.check_known_game_rarity(request.question)
        if known_tier:
            context += f"\n\n---\n\nKnown in-game rarity tier (Steal a Brainrot): {known_tier}"

    user_message = f"Context from the Italian Brainrot wiki:\n\n{context}\n\nRequest: {request.question}\n\n{mode['instruction']}"

    response = handle_gemini_errors(
        lambda: gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
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
