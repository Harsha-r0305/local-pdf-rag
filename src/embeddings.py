from sentence_transformers import SentenceTransformer

from document_loader import load_pdf
from chunker import create_chunks


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    pdf_path = "data/documents/paper.pdf"

    documents = load_pdf(pdf_path)

    chunks = create_chunks(documents)

    print(f"Total chunks: {len(chunks)}")

    embeddings = create_embeddings(chunks)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\nFirst embedding:")
    print(embeddings[0])