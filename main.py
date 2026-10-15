"""PDF RAG system - command line entry point.

Usage
-----
    python main.py build                      # (re)build the vector index
    python main.py build --splitter recursive # build with the word-safe splitter
    python main.py ask "What is this paper about?" # one-shot question
    python main.py chat                       # interactive Q&A loop
    python main.py eval                       # retrieval evaluation (fast)
    python main.py eval --generate            # + answer evaluation (needs Ollama)

Pipeline
--------
    PDF -> document_loader -> chunking -> MiniLM embeddings -> FAISS index
    question -> embed -> FAISS top-k -> prompt -> Ollama (qwen2.5:3b) -> answer
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# The modules in src/ import each other with flat imports
# (e.g. `from document_loader import load_pdf`), so put src/ on the path.
sys.path.insert(0, str(SRC_DIR))

INDEX_FILE = PROJECT_ROOT / "vectorstore" / "index.faiss"
CHUNKS_FILE = PROJECT_ROOT / "vectorstore" / "chunks.pkl"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def index_exists():
    return INDEX_FILE.exists() and CHUNKS_FILE.exists()


def require_index():
    if not index_exists():
        sys.exit(
            "No vector index found.\n"
            "Build it first with:  python main.py build"
        )


def explain_llm_error(error):
    """Turn raw Ollama failures into an actionable message."""
    text = str(error).lower()

    if "connect" in text or "refused" in text:
        return (
            "Could not reach Ollama. Make sure it is running "
            "(start the Ollama app, or run `ollama serve`)."
        )

    if "not found" in text or "pull" in text:
        return (
            "The Ollama model is missing. Download it with:\n"
            "    ollama pull qwen2.5:3b"
        )

    return f"LLM error: {error}"


def print_answer(answer, results):
    print(f"\nAI: {answer}\n")

    if results:
        print("Sources used (page, similarity):")
        for result in results:
            print(f"  - {result['source']}  p.{result['page']}  "
                  f"score={result['score']:.3f}")
        print()


# --------------------------------------------------
# Commands
# --------------------------------------------------

def cmd_build(args):
    from vector_store import build_index

    build_index(splitter=args.splitter)


def cmd_ask(args):
    require_index()

    from rag import ask_rag

    try:
        answer, results = ask_rag(
            args.question, top_k=args.top_k, return_context=True
        )
    except Exception as error:
        sys.exit(explain_llm_error(error))

    print_answer(answer, results)


def cmd_chat(args):
    require_index()

    from rag import ask_rag

    print("=" * 60)
    print("              PDF RAG SYSTEM")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer, results = ask_rag(
                question, top_k=args.top_k, return_context=True
            )
            print_answer(answer, results)
        except Exception as error:
            print(f"\n{explain_llm_error(error)}\n")


def cmd_eval(args):
    require_index()

    sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
    from rag_eval import run_from_cli

    run_from_cli(k=args.k, generate=args.generate)


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="PDF question answering with retrieval-augmented generation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build the FAISS vector index.")
    p_build.add_argument(
        "--splitter",
        choices=["sliding", "recursive"],
        default="sliding",
        help="Chunking strategy (default: sliding, the original behaviour).",
    )
    p_build.set_defaults(func=cmd_build)

    p_ask = sub.add_parser("ask", help="Ask a single question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=6)
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive question loop.")
    p_chat.add_argument("--top-k", type=int, default=6)
    p_chat.set_defaults(func=cmd_chat)

    p_eval = sub.add_parser("eval", help="Evaluate retrieval (and answers).")
    p_eval.add_argument("--k", type=int, default=5, help="Chunks to retrieve.")
    p_eval.add_argument(
        "--generate",
        action="store_true",
        help="Also generate answers with Ollama and score them (slow).",
    )
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
