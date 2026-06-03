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


class ImproveExplanation(dspy.Signature):
    """
    Improve a rough technical explanation for a software engineering audience.

    Preserve the user's original meaning, but make it clearer, more precise,
    and easier to say in an interview or teaching context.
    """

    rough_explanation: str = dspy.InputField()
    audience: str = dspy.InputField(desc="Target audience, e.g. junior engineer, staff engineer, PM")
    improved_explanation: str = dspy.OutputField()
    key_changes: str = dspy.OutputField(desc="Briefly explain what was improved")


def main():
    configure_dspy()

    improve = dspy.Predict(ImproveExplanation)

    result = improve(
        rough_explanation=(
            "Caching is when you save data so it's faster later, "
            "but it can be wrong sometimes if the real data changes."
        ),
        audience="junior backend engineer",
    )

    print("\nImproved explanation:\n")
    print(result.improved_explanation)

    print("\nKey changes:\n")
    print(result.key_changes)



if __name__ == "__main__":
    main()
