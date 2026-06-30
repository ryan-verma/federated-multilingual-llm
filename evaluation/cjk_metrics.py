"""
Character-level EM/F1 for Chinese (and other non-whitespace-delimited
languages, if ever needed).

evaluate.load("squad") computes F1 by splitting predicted/gold text on
WHITESPACE and comparing token overlap. Chinese doesn't use whitespace
between words, so the whole string is treated as one long pseudo-token --
this collapses partial credit almost entirely and makes F1 collapse down
toward EM (the symptom you'd see: F1 barely higher than EM, when normally
F1 is well above EM since it gives partial credit).

This mirrors the official MLQA evaluation script's handling: characters
are used as the comparison unit for Chinese instead of whitespace-split
words. Everything else (lowercasing, punctuation/article removal) follows
the same normalization SQuAD-style metrics use.
"""

import re
import string
import collections


def normalize_answer(text):
    """Standard SQuAD-style normalization: lowercase, remove punctuation,
    remove articles, collapse whitespace. Same normalization used by the
    official squad metric, applied here before tokenizing."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s):
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def char_tokenize(text):
    """Tokenize by individual character (after normalization), dropping
    whitespace -- the right comparison unit for Chinese, where words
    aren't whitespace-delimited."""
    normalized = normalize_answer(text)
    return [ch for ch in normalized if not ch.isspace()]


def compute_exact(gold_text, pred_text):
    return int(normalize_answer(gold_text) == normalize_answer(pred_text))


def compute_f1_char_level(gold_text, pred_text):
    gold_toks = char_tokenize(gold_text)
    pred_toks = char_tokenize(pred_text)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is empty, F1 is 1 only if both are empty.
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_cjk_metrics(formatted_predictions, references):
    """Drop-in replacement for squad_metric.compute() output shape
    ({"exact_match": ..., "f1": ...}), but using character-level F1.
    `formatted_predictions` / `references` follow the same format
    evaluate_mlqa.py already builds: [{"id", "prediction_text"}] and
    [{"id", "answers": {"text": [...]}}]."""

    pred_by_id = {p["id"]: p["prediction_text"] for p in formatted_predictions}

    exact_scores = []
    f1_scores = []

    for ref in references:
        example_id = ref["id"]
        pred_text = pred_by_id.get(example_id, "")
        gold_texts = ref["answers"]["text"]

        if not gold_texts:
            continue

        # Take the best score across all acceptable gold answers, same as
        # the official SQuAD/MLQA evaluation approach.
        exact_scores.append(max(compute_exact(g, pred_text) for g in gold_texts))
        f1_scores.append(max(compute_f1_char_level(g, pred_text) for g in gold_texts))

    return {
        "exact_match": 100.0 * sum(exact_scores) / len(exact_scores),
        "f1": 100.0 * sum(f1_scores) / len(f1_scores),
    }