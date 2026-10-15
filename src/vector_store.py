"""Build the FAISS vector store from the PDF in data/documents/.

Run directly:      python src/vector_store.py
Or via main.py:    python main.py build

Nothing runs at import time any more - call build_index() explicitly.
"""

import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from document_loader import load_pdf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Project root: RAG_P1/RAG_P1
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "data" / "documents" / "paper.pdf"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# --------------------------------------------------
# Chunking strategies
# --------------------------------------------------

def create_sliding_window_chunks(documents, chunk_size=CHUNK_SIZE,
                                 chunk_overlap=CHUNK_OVERLAP):
    """Fixed-width character windows (the original strategy).

    Simple and fast, but it cuts text at an exact character count, so many
    chunks start or end in the middle of a word or sentence.
    """
    chunks = []

    for document in documents:
        text = " ".join(document["text"].split())

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": document["source"],
                    "page": document["page"],
                })

            start += chunk_size - chunk_overlap

    return chunks


def create_recursive_chunks(documents):
    """Sentence/paragraph-aware splitting using the existing chunker.py."""
    from chunker import create_chunks

    return create_chunks(documents)


SPLITTERS = {
    "sliding": create_sliding_window_chunks,
    "recursive": create_recursive_chunks,
}


# --------------------------------------------------
# Build
# --------------------------------------------------

def build_index(splitter="sliding", pdf_path=PDF_PATH):
    """Load the PDF, chunk it, embed the chunks and save a FAISS index."""

    if splitter not in SPLITTERS:
        raise ValueError(
            f"Unknown splitter '{splitter}'. Choose from: {list(SPLITTERS)}"
        )

    print("Loading PDF...")
    documents = load_pdf(str(pdf_path))
    print(f"Pages loaded: {len(documents)}")

    print(f"Creating chunks (splitter = {splitter})...")
    chunks = SPLITTERS[splitter](documents)
    print(f"Chunks created: {len(chunks)}")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]

    # Normalised vectors + inner-product index == cosine similarity.
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Embedding shape: {embeddings.shape}")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Vectors stored: {index.ntotal}")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving vector database...")
    faiss.write_index(index, str(INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print("\nVector database created successfully!")
    print(f"Index:  {INDEX_PATH}")
    print(f"Chunks: {CHUNKS_PATH}")

    return index, chunks


if __name__ == "__main__":
    build_index()
