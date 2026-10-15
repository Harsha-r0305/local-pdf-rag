"""Evaluate the RAG system on evaluation/questions.json.

Two layers are measured:

1. RETRIEVAL (fast, no LLM needed)
   Did the right chunk come back from FAISS?
     Hit@k      - a gold chunk appears somewhere in the top-k
     MRR        - 1 / rank of the first gold chunk (higher = ranked earlier)
     PageHit@k  - looser: any chunk from a gold page appears in the top-k
   A chunk is "gold" if it is on one of the question's gold_pages AND
   contains at least one of its retrieval_keywords.

2. ANSWERS (--generate, needs Ollama running)
   Keyword recall : share of answer_keywords present in the answer
   Citation rate  : does the answer mention a page number
   Refusal rate   : on questions the paper cannot answer, does the model
                    say so instead of making something up

Run:
    python main.py eval
    python main.py eval --generate
"""

import json
import re
import statistics
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.json"

REFUSAL_PATTERNS = [
    r"could not find",
    r"couldn'?t find",
    r"cannot find",
    r"can'?t find",
    r"not (?:mentioned|stated|specified|provided|discussed|included)",
    r"no (?:relevant )?information",
    r"does not (?:contain|mention|provide|specify|state|say)",
    r"doesn'?t (?:contain|mention|provide|specify|state|say)",
]

CITATION_PATTERN = re.compile(r"\bpages?\s*\d|\bp\.\s*\d|\bpp\.\s*\d", re.I)


# --------------------------------------------------
# Matching helpers
# --------------------------------------------------

def normalize(text):
    """Lower-case and drop whitespace/hyphens, so PDF line-wrap artefacts
    such as 'misconfigu- ration' do not break keyword matching."""
    return re.sub(r"[\s\-]+", "", text.lower())


def contains_any(text, keywords):
    haystack = normalize(text)
    return any(normalize(keyword) in haystack for keyword in keywords)


def is_gold_chunk(chunk, question):
    return (
        chunk["page"] in question["gold_pages"]
        and contains_any(chunk["text"], question["retrieval_keywords"])
    )


def is_gold_page(chunk, question):
    return chunk["page"] in question["gold_pages"]


def first_gold_rank(chunks, question, predicate):
    """1-based rank of the first chunk matching predicate, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if predicate(chunk, question):
            return rank
    return None


def keyword_recall(answer, keywords):
    if not keywords:
        return None
    haystack = normalize(answer)
    found = sum(1 for keyword in keywords if normalize(keyword) in haystack)
    return found / len(keywords)


def is_refusal(answer):
    text = answer.lower()
    return any(re.search(pattern, text) for pattern in REFUSAL_PATTERNS)


# --------------------------------------------------
# Core evaluation (pure: callables are injected, so it is easy to test)
# --------------------------------------------------

def evaluate(questions, retrieve_fn, k=5, ask_fn=None):
    """Score the system.

    retrieve_fn(query, k) -> list of {"text","page","score",...}
    ask_fn(question)      -> (answer, results)    [optional, for --generate]
    """
    rows = []

    for question in questions:
        results = retrieve_fn(question["question"], k)
        answerable = question["type"] != "unanswerable"

        row = {
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "top1_score": results[0]["score"] if results else None,
            "retrieved_pages": [r["page"] for r in results],
        }

        if answerable:
            row["gold_rank"] = first_gold_rank(results, question, is_gold_chunk)
            row["page_rank"] = first_gold_rank(results, question, is_gold_page)

        if ask_fn is not None:
            try:
                answer, _ = ask_fn(question["question"])
                row["answer"] = answer
                if answerable:
                    row["keyword_recall"] = keyword_recall(
                        answer, question["answer_keywords"]
                    )
                    row["cited_page"] = bool(CITATION_PATTERN.search(answer))
                else:
                    row["refused"] = is_refusal(answer)
            except Exception as error:
                row["answer_error"] = str(error)

        rows.append(row)

    return rows, summarize(rows, k)


def summarize(rows, k):
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]

    def hit_at(n):
        return sum(1 for r in answerable
                   if r["gold_rank"] is not None and r["gold_rank"] <= n) \
            / max(len(answerable), 1)

    def page_hit_at(n):
        return sum(1 for r in answerable
                   if r["page_rank"] is not None and r["page_rank"] <= n) \
            / max(len(answerable), 1)

    mrr = statistics.mean(
        (1 / r["gold_rank"]) if r["gold_rank"] else 0.0 for r in answerable
    ) if answerable else 0.0

    summary = {
        "k": k,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        f"hit@{k}": hit_at(k),
        f"page_hit@{k}": page_hit_at(k),
        "mrr": mrr,
        "mean_top1_score_answerable": _mean(r["top1_score"] for r in answerable),
        "mean_top1_score_unanswerable": _mean(r["top1_score"] for r in unanswerable),
    }

    scored = [r for r in answerable if r.get("keyword_recall") is not None]
    if scored:
        summary["answer_keyword_recall"] = _mean(r["keyword_recall"] for r in scored)
        summary["answer_fully_correct"] = (
            sum(1 for r in scored if r["keyword_recall"] == 1.0) / len(scored)
        )
        summary["citation_rate"] = (
            sum(1 for r in scored if r.get("cited_page")) / len(scored)
        )

    refusals = [r for r in unanswerable if "refused" in r]
    if refusals:
        summary["refusal_rate_unanswerable"] = (
            sum(1 for r in refusals if r["refused"]) / len(refusals)
        )

    return summary


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


# --------------------------------------------------
# Reporting
# --------------------------------------------------

def print_report(rows, summary):
    k = summary["k"]
    has_answers = any("answer" in r or "answer_error" in r for r in rows)

    print("\n" + "=" * 78)
    print("PER-QUESTION RESULTS".center(78))
    print("=" * 78)
    header = f"{'ID':<5}{'type':<13}{'gold rank':<11}{'page rank':<11}{'top1 sim':<10}"
    if has_answers:
        header += "answer"
    print(header)
    print("-" * 78)

    for r in rows:
        gold = r.get("gold_rank")
        page = r.get("page_rank")
        gold_s = "n/a" if r["type"] == "unanswerable" else (str(gold) if gold else "MISS")
        page_s = "n/a" if r["type"] == "unanswerable" else (str(page) if page else "MISS")
        top1 = f"{r['top1_score']:.3f}" if r["top1_score"] is not None else "-"

        line = f"{r['id']:<5}{r['type']:<13}{gold_s:<11}{page_s:<11}{top1:<10}"

        if has_answers:
            if "answer_error" in r:
                line += "ERROR"
            elif r["type"] == "unanswerable":
                line += "refused (good)" if r.get("refused") else "ANSWERED (bad)"
            else:
                line += f"recall {r['keyword_recall']:.2f}"
        print(line)

    print("\n" + "=" * 78)
    print("SUMMARY".center(78))
    print("=" * 78)
    print(f"Answerable questions : {summary['n_answerable']}")
    print(f"Unanswerable (traps) : {summary['n_unanswerable']}\n")
    print("Retrieval")
    print(f"  Hit@1            : {summary['hit@1']:.1%}")
    print(f"  Hit@3            : {summary['hit@3']:.1%}")
    print(f"  Hit@{k:<13}: {summary[f'hit@{k}']:.1%}")
    print(f"  PageHit@{k:<9}: {summary[f'page_hit@{k}']:.1%}   (looser: right page, any chunk)")
    print(f"  MRR              : {summary['mrr']:.3f}")

    a = summary["mean_top1_score_answerable"]
    u = summary["mean_top1_score_unanswerable"]
    if a is not None and u is not None:
        print(f"  Mean top-1 sim   : answerable {a:.3f}  vs  unanswerable {u:.3f}")
        print(f"                     gap = {a - u:+.3f}  (a bigger gap means a")
        print("                     similarity threshold could reject off-topic questions)")

    if "answer_keyword_recall" in summary:
        print("\nAnswers")
        print(f"  Keyword recall   : {summary['answer_keyword_recall']:.1%}")
        print(f"  Fully correct    : {summary['answer_fully_correct']:.1%}")
        print(f"  Cites a page     : {summary['citation_rate']:.1%}")
    if "refusal_rate_unanswerable" in summary:
        print(f"  Refuses traps    : {summary['refusal_rate_unanswerable']:.1%}")

    misses = [r["id"] for r in rows if r["type"] != "unanswerable" and not r.get("gold_rank")]
    if misses:
        print(f"\nRetrieval misses (no gold chunk in top-{k}): {', '.join(misses)}")
    print()


# --------------------------------------------------
# Wiring to the real system
# --------------------------------------------------

def load_questions(path=QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_from_cli(k=5, generate=False):
    from retriever import retrieve  # loads MiniLM + FAISS (needs the index)

    def retrieve_fn(query, k_):
        return retrieve(query, k=k_)

    ask_fn = None
    if generate:
        from rag import ask_rag

        def ask_fn(question):
            return ask_rag(question, top_k=6, return_context=True)

    questions = load_questions()
    rows, summary = evaluate(questions, retrieve_fn, k=k, ask_fn=ask_fn)

    print_report(rows, summary)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "rows": rows}, file, indent=2)

    print(f"Detailed results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    import sys

    # Allow `python evaluation/rag_eval.py` as well as `python main.py eval`.
    sys.path.insert(0, str(EVAL_DIR.parent / "src"))
    run_from_cli()
