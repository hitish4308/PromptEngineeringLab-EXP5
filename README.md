# Prompt Engineering Lab - Experiment 5

## Zero-Shot vs Few-Shot Classification

Compares zero-shot and few-shot prompting on a sentiment
classification task, measuring accuracy and format adherence.

## Task

Classify customer reviews as Positive, Neutral, or Negative.

## Test Set

Five hardcoded reviews with ground-truth labels, spanning clear
positive, negative, and neutral cases.

## Methods

| Method | Description |
|--------|-------------|
| Zero-shot | No examples - just the instruction and target review |
| Few-shot | 3 labelled examples preceding the target review |

## Model and Parameters

- Model: openai/gpt-oss-20b
- Temperature: 0.0 (identical for both methods)
- Max tokens: 20 (each answer must be one word)

## Setup

Reuse the environment from Experiment 1:

    Copy-Item ..\EXP_1\.env .
    python -m pip install -r requirements.txt

## Run

    python classify_reviews.py

## Expected Output

For each review:

- Ground truth
- Zero-shot prediction
- Few-shot prediction

Followed by a comparison table with per-review correctness and
aggregate accuracy for each method.

## Key Findings

- Few-shot prompting supplies in-context examples that the model
  mimics, enforcing both label vocabulary and output format
- Zero-shot relies on pretrained understanding - often accurate on
  clear cases, may drift on mixed or sarcastic reviews
- Using temperature=0 ensures deterministic results
- Both methods benefit from a strict single-word output constraint

## When to Use Which

| Situation | Recommendation |
|-----------|----------------|
| Output format matters | Few-shot |
| Speed and low token cost | Zero-shot |
| Ambiguous or domain-specific task | Few-shot |
| Broad, well-understood task | Zero-shot |

## Security

- Never commit .env
- Never paste API keys in chat, logs, or screenshots

## License

For educational / lab use only.
