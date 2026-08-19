#!/usr/bin/env python3
"""Chunk wiki_data.json, embed the chunks locally, and store them in Chroma."""

import json

import chromadb
from sentence_transformers import SentenceTransformer

INPUT_FILE = "wiki_data.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "wiki_pages"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHUNK_TOKENS = 500
EMBED_BATCH_SIZE = 64
ADD_BATCH_SIZE = 500

model = SentenceTransformer(EMBEDDING_MODEL)


def count_tokens(text):
    return len(model.tokenizer.encode(text, add_special_tokens=False))


def split_by_tokens(text, max_tokens):
    """Hard-split text into pieces of at most max_tokens tokens."""
    tokens = model.tokenizer.encode(text, add_special_tokens=False)
    pieces = []
    for start in range(0, len(tokens), max_tokens):
        piece_tokens = tokens[start : start + max_tokens]
        pieces.append(model.tokenizer.decode(piece_tokens))
    return pieces


def group_by_tokens(units, max_tokens):
    """Greedily accumulate text units (paragraphs/sentences) into ~max_tokens chunks."""
    chunks = []
    current, current_tokens = [], 0

    for unit in units:
        unit_tokens = count_tokens(unit)

        if unit_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            sentences = unit.replace("\n", " ").split(". ")
            chunks.extend(group_by_tokens(sentences, max_tokens) if len(sentences) > 1 else split_by_tokens(unit, max_tokens))
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_page(content, max_tokens=MAX_CHUNK_TOKENS):
    content = content.strip()
    if not content:
        return []
    if count_tokens(content) <= max_tokens:
        return [content]

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return group_by_tokens(paragraphs, max_tokens)


def build_chunks(pages):
    chunks, metadatas, ids = [], [], []
    for page_idx, page in enumerate(pages):
        for chunk_idx, chunk in enumerate(chunk_page(page["content"])):
            chunks.append(chunk)
            metadatas.append({"title": page["title"], "url": page["url"]})
            ids.append(f"{page_idx}_{chunk_idx}")
    return chunks, metadatas, ids


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages from {INPUT_FILE}")

    chunks, metadatas, ids = build_chunks(pages)
    print(f"Chunked into {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION_NAME)

    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        batch_chunks = chunks[start : start + ADD_BATCH_SIZE]
        batch_metadatas = metadatas[start : start + ADD_BATCH_SIZE]
        batch_ids = ids[start : start + ADD_BATCH_SIZE]

        embeddings = model.encode(batch_chunks, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False).tolist()
        collection.add(ids=batch_ids, embeddings=embeddings, documents=batch_chunks, metadatas=batch_metadatas)
        print(f"Indexed {min(start + ADD_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} chunks in {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
