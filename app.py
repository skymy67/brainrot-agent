#!/usr/bin/env python3
"""FastAPI backend that answers questions via RAG over the Italian Brainrot wiki."""

import chromadb
from fastapi import FastAPI
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

app = FastAPI(title="Italian Brainrot Wiki Chat")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
gemini_client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment


class ChatRequest(BaseModel):
    question: str


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
    documents, metadatas = retrieve_chunks(request.question)

    context = "\n\n---\n\n".join(f"[{meta['title']}]\n{doc}" for doc, meta in zip(documents, metadatas))
    user_message = (
        f"Context from the Italian Brainrot wiki:\n\n{context}\n\n"
        f"Question: {request.question}\n\n"
        "Answer the question using only the context above. "
        "If the context doesn't contain the answer, say so."
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant answering questions about the Italian Brainrot wiki using the provided context.",
            max_output_tokens=1024,
        ),
    )
    answer = response.text or ""

    return ChatResponse(answer=answer, sources=dedupe_sources(metadatas))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
