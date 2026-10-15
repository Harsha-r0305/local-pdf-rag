from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_pdf


def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for document in documents:

        split_texts = text_splitter.split_text(document["text"])

        for text in split_texts:

            chunks.append({
                "text": text,
                "source": document["source"],
                "page": document["page"]
            })

    return chunks


if __name__ == "__main__":

    pdf_path = "data/documents/paper.pdf"

    documents = load_pdf(pdf_path)

    chunks = create_chunks(documents)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):

        print("\n" + "=" * 60)
        print(f"Chunk: {i + 1}")
        print(f"Source: {chunk['source']}")
        print(f"Page: {chunk['page']}")
        print("=" * 60)

        print(chunk["text"])