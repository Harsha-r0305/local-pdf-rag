from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "text": text.strip(),
                "source": Path(pdf_path).name,
                "page": page_number
            })

    return documents


if __name__ == "__main__":

    pdf_path = "data/documents/paper.pdf"

    documents = load_pdf(pdf_path)

    print(f"Total pages loaded: {len(documents)}")

    for document in documents[:3]:

        print("\n" + "=" * 60)
        print(f"Source: {document['source']}")
        print(f"Page: {document['page']}")
        print("=" * 60)

        print(document["text"][:1000])