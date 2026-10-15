import ollama


MODEL_NAME = "qwen2.5:3b"


def ask_llm(question):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    question = input("Ask the AI something: ")

    answer = ask_llm(question)

    print("\nAI Answer:")
    print(answer)