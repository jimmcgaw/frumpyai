import os

import dspy
from dotenv import load_dotenv


def configure_dspy() -> dspy.LM:
    """
    Configure DSPy to use a locally running Ollama model.

    LiteLLM supports Ollama model strings like:
        ollama_chat/llama3.2:3b

    Ollama must be running separately.
    """
    load_dotenv()

    model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    lm = dspy.LM(
        model=f"ollama_chat/{model}",
        api_base=base_url,
        temperature=0.2,
        max_tokens=500,
    )

    dspy.configure(lm=lm)
    return lm


def main():
    configure_dspy()

    improve_explanation = dspy.Predict(
        "rough_explanation -> improved_explanation"
    )

    result = improve_explanation(
        rough_explanation=(
            "Idempotency means if you retry something it doesn't mess things up. "
            "It's useful when networks fail."
        )
    )

    print("\nImproved explanation:\n")
    print(result.improved_explanation)

    print("\n--- DSPy generated prompt/history ---\n")
    dspy.inspect_history(n=1)



if __name__ == "__main__":
    main()
