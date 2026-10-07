"""
Experiment 5 - Zero-Shot vs Few-Shot Classification

Task: Classify customer reviews as Positive, Neutral, or Negative.
Compares accuracy and format adherence between:
  - Zero-shot prompt (no examples)
  - Few-shot prompt (3 examples before the target)

Uses identical model and parameters (temperature=0) for both.
"""

import os
import re
import textwrap
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Setup ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise SystemExit("NVIDIA_API_KEY not found. Create a .env file.")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.0
MAX_TOKENS = 20


# ---------- Test Set (with ground truth) ----------
TEST_REVIEWS = [
    ("The battery life is amazing!", "Positive"),
    ("It works, nothing special.", "Neutral"),
    ("Worst purchase of my life.", "Negative"),
    ("The camera is slow but the screen is gorgeous.", "Neutral"),
    ("Absolutely love it, will buy again!", "Positive"),
]


# ---------- Prompt Templates ----------
ZERO_SHOT_TEMPLATE = (
    "Classify the following review as Positive, Neutral, or Negative.\n"
    "Answer with one word only: Positive, Neutral, or Negative.\n\n"
    "Review: {review}\n"
    "Sentiment:"
)

FEW_SHOT_TEMPLATE = (
    "Classify each review as Positive, Neutral, or Negative.\n"
    "Answer with one word only.\n\n"
    "Review: \"Best phone ever!\" -> Positive\n"
    "Review: \"The camera is slow.\" -> Negative\n"
    "Review: \"Average quality, nothing special.\" -> Neutral\n\n"
    "Review: \"{review}\" ->"
)


# ---------- Helpers ----------
def ask_llm(prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except Exception as e:
        return f"[API error] {type(e).__name__}: {e}"


def normalize(text: str) -> str:
    """Extract Positive / Neutral / Negative from the model's response."""
    text_lower = text.lower()
    for label in ("positive", "neutral", "negative"):
        if label in text_lower:
            return label.capitalize()
    return text.strip() or "?"


def run_zero_shot(review: str) -> str:
    prompt = ZERO_SHOT_TEMPLATE.format(review=review)
    raw = ask_llm(prompt)
    return normalize(raw)


def run_few_shot(review: str) -> str:
    prompt = FEW_SHOT_TEMPLATE.format(review=review)
    raw = ask_llm(prompt)
    return normalize(raw)


# ---------- Main ----------
if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE} | Max tokens: {MAX_TOKENS}")
    print()

    results = []

    print("=" * 72)
    print("  RUNNING CLASSIFICATION")
    print("=" * 72)

    for i, (review, truth) in enumerate(TEST_REVIEWS, start=1):
        print(f"\nReview {i}: {review}")
        print(f"  Ground truth : {truth}")

        zs = run_zero_shot(review)
        print(f"  Zero-shot    : {zs}")

        fs = run_few_shot(review)
        print(f"  Few-shot     : {fs}")

        results.append({
            "n": i,
            "review": review,
            "truth": truth,
            "zero_shot": zs,
            "few_shot": fs,
            "zs_correct": zs == truth,
            "fs_correct": fs == truth,
        })

    # ---------- Comparison Table ----------
    print("\n" + "=" * 72)
    print("  COMPARISON TABLE")
    print("=" * 72)
    print(f"{'#':<3}{'Ground Truth':<13}{'Zero-Shot':<13}{'Few-Shot':<13}{'ZS ok':<7}{'FS ok':<7}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['n']:<3}"
            f"{r['truth']:<13}"
            f"{r['zero_shot']:<13}"
            f"{r['few_shot']:<13}"
            f"{'Y' if r['zs_correct'] else 'N':<7}"
            f"{'Y' if r['fs_correct'] else 'N':<7}"
        )
    print("-" * 72)

    # ---------- Accuracy ----------
    zs_hits = sum(1 for r in results if r["zs_correct"])
    fs_hits = sum(1 for r in results if r["fs_correct"])
    total = len(results)

    print("\nACCURACY:")
    print(f"  Zero-shot : {zs_hits}/{total} = {zs_hits/total:.0%}")
    print(f"  Few-shot  : {fs_hits}/{total} = {fs_hits/total:.0%}")

    # ---------- Summary ----------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print("""
Findings:
  - Few-shot prompting supplies in-context examples, which the model
    mimics. This tends to improve accuracy on ambiguous cases and
    enforces consistent output formatting.
  - Zero-shot relies on the model's pretrained understanding. It often
    performs well on clear-cut cases but may drift on mixed or
    sarcastic reviews.
  - Both methods should use identical temperature and max_tokens for
    a fair comparison. Temperature=0 minimizes randomness.

Takeaways:
  - Few-shot is preferred when the output format matters.
  - Zero-shot is preferred for speed and lower token cost.
  - The gap between the two shrinks as model quality increases.
""")
