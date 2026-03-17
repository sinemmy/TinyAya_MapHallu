# Answer Match Rate (AMR) computation.

import re


def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def answer_match(response: str | None, ground_truth: str | None) -> float | None:
    """
    Scoring logic:
      1.0  — ground truth appears anywhere in the response as a substring.
              Handles generative answers that embed the correct answer in a
              full sentence e.g. "The capital of France is Paris."
      0–1  — token-level overlap (F1-style) if no substring match found.
              Partial credit for responses that get some tokens right.
      0.0  — no overlap at all.
      None — ground truth is missing, cannot score this row.
    """
    if ground_truth is None:
        return None
    if not response:
        return 0.0

    r = normalize(response)
    g = normalize(ground_truth)

    if not g:
        return None

    if g in r:
        return 1.0

    r_tokens = set(r.split())
    g_tokens = set(g.split())
    overlap = len(r_tokens & g_tokens)
    return round(overlap / len(g_tokens), 4)


def batch_amr(responses: list[str | None], ground_truth: str | None) -> list[float | None]:
    return [answer_match(r, ground_truth) for r in responses]


def mean_amr(responses: list[str | None], ground_truth: str | None) -> float | None:
    scores = [s for s in batch_amr(responses, ground_truth) if s is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)