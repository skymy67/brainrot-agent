#!/usr/bin/env python3
"""FastAPI backend that answers questions via RAG over the Italian Brainrot wiki."""

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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
        "max_output_tokens": 1024,
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
        "max_output_tokens": 2048,
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
        "max_output_tokens": 1536,
    },
}

app = FastAPI(title="Italian Brainrot Wiki Chat")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment


class ChatRequest(BaseModel):
    question: str
    mode: str = "qa"


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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{request.mode}'. Use 'qa' or 'creative'.")
    mode = MODES[request.mode]

    documents, metadatas = retrieve_chunks(request.question)

    context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))
    user_message = f"Context from the Italian Brainrot wiki:\n\n{context}\n\nRequest: {request.question}\n\n{mode['instruction']}"

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=mode["system_instruction"],
            max_output_tokens=mode["max_output_tokens"],
        ),
    )
    answer = response.text or ""

    return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))


# Serves static/index.html at "/" — registered after /chat so the API route takes priority.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
