
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "vectorstore" / "index.faiss"
CHUNKS_PATH = PROJECT_ROOT / "vectorstore" / "chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))

print("Loading chunks...")
with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)

if index.ntotal != len(chunks):
    raise ValueError(
        f"FAISS index contains {index.ntotal} vectors, "
        f"but {len(chunks)} chunks were loaded."
    )

print(f"Retriever ready: {len(chunks)} chunks loaded.")


def retrieve(query, k=5, top_k=None):
    """Return relevant chunks along with their source and page metadata."""

    if top_k is not None:
        k = top_k

    if not query or not query.strip():
        return []

    k = min(k, index.ntotal)

    if k <= 0:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    distances, indices = index.search(query_embedding, k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        # Support both dictionary chunks and older string-only chunks.
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            source = chunk.get("source", "paper.pdf")
            page = chunk.get("page", "Unknown")
        else:
            text = str(chunk)
            source = "paper.pdf"
            page = "Unknown"

        results.append({
            "text": text,
            "source": source,
            "page": page,
            "score": float(distance)
        })

    return results


if __name__ == "__main__":
    query = input("\nAsk a question: ")
    results = retrieve(query, k=3)

    print("\nRetrieved Documents:\n")

    for i, result in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])
        print()
