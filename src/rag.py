
import ollama

from retriever import retrieve

MODEL_NAME = "qwen2.5:3b"


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"[Document Chunk {i}]\n"
            f"Source: {result['source']}\n"
            f"Page: {result['page']}\n"
            f"Relevance Score: {result['score']:.4f}\n"
            f"Content:\n{result['text']}"
        )

    return "\n\n".join(context_parts)


def ask_rag(question, top_k=6, return_context=False):
    """Answer a question from the indexed document.

    Returns the answer string. If return_context=True, returns a tuple
    (answer, results) where results are the retrieved chunks - used by the
    CLI to show sources and by the evaluation script.
    """
    # Retrieve more chunks to give the model better context.
    results = retrieve(question, top_k=top_k)

    if not results:
        answer = "I could not find relevant information in the document."
        return (answer, results) if return_context else answer

    context = build_context(results)

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the supplied document context.

Rules:
1. Use the context as your evidence.
2. Do not invent facts or use outside knowledge.
3. For questions about the paper's title, authors, abstract,
   problem, or proposed solution, prioritize context that directly
   discusses that topic.
4. If the context does not contain the answer, say:
   "I could not find this information in the document."
5. Answer clearly and explain technical terms when appropriate.
6. Cite the relevant page number or numbers in your answer.

DOCUMENT CONTEXT
================
{context}
================

QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return (answer, results) if return_context else answer


if __name__ == "__main__":
    print("=" * 60)
    print("              PDF RAG SYSTEM")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = ask_rag(question)
            print("\nAI:", answer, "\n")
        except Exception as error:
            print(f"\nError: {error}\n")
