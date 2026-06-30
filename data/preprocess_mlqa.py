"""
Preprocessing for the centralized MLQA baseline.

This file builds the per-language federated client train/validation splits
that training/train_centralized.py pools together. Real EM/F1 evaluation,
broken out per language, lives separately in evaluation/evaluate_mlqa.py --
that script already builds its own eval data with offset_mapping/example_id
preserved (via evaluation/preprocess_eval.py) and decodes spans via
evaluation/metrics.py, so that machinery doesn't need to be duplicated here.

The "validation" split produced below is only used to track eval LOSS during
training (a normal convergence signal, via start/end label positions) -- not
EM/F1. That keeps this file simple and avoids two different implementations
of the same metric existing in the project at once.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 384
STRIDE = 128

# NOTE: MLQA's translate-train split does not include an "en" config --
# translate-train is, by definition, SQuAD translated INTO other languages,
# so there's no "translated into English" version. The "en" client uses
# plain English SQuAD instead, which shares MLQA's schema
# (id / question / context / answers), so it needs no special handling
# anywhere else in this file.
TRAIN_CONFIGS = {
    "hi": "mlqa-translate-train.hi",
    "es": "mlqa-translate-train.es",
    "de": "mlqa-translate-train.de",
    "zh": "mlqa-translate-train.zh",
}

EVAL_CONFIGS = {
    "en": "mlqa.en.en",
    "hi": "mlqa.hi.hi",
    "es": "mlqa.es.es",
    "de": "mlqa.de.de",
    "zh": "mlqa.zh.zh",
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_train_split(lang):
    """English uses SQuAD; every other language uses its MLQA translate-train split."""
    if lang == "en":
        return load_dataset("squad")["train"]
    return load_dataset("facebook/mlqa", TRAIN_CONFIGS[lang])["train"]


def preprocess_examples(examples):
    """Tokenizes examples and converts answers into integer start/end token
    positions. Used for both train and validation -- both need labels for
    loss; text-level EM/F1 decoding is a separate concern handled in
    evaluation/."""
    questions = [q.strip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_offsets_mapping=True,
        padding="max_length",
        return_overflowing_tokens=True,
    )

    offset_mapping = tokenized.pop("offset_mapping")
    overflow_to_sample_mapping = tokenized.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = overflow_to_sample_mapping[i]
        answer = examples["answers"][sample_index]

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start = context_start
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            token_end = context_end
            while token_end >= 0 and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def load_single_client_dataset(lang):
    """Builds train/validation data for one language only. Used by federated
    clients, where each client only ever needs its own language -- unlike
    prepare_client_datasets() below, which builds all five and is meant for
    the centralized baseline that pools every language together."""
    raw_train = load_train_split(lang)
    train_dataset = raw_train.map(
        preprocess_examples,
        batched=True,
        remove_columns=raw_train.column_names,
    )

    raw_validation = load_dataset("facebook/mlqa", EVAL_CONFIGS[lang])["validation"]
    validation_dataset = raw_validation.map(
        preprocess_examples,
        batched=True,
        remove_columns=raw_validation.column_names,
    )

    return {
        "train": train_dataset,
        "validation": validation_dataset,
    }


def prepare_client_datasets():
    """Builds per-language federated client datasets: tokenized train data,
    plus a tokenized validation split (MLQA's native dev set, never used for
    training) for loss tracking during centralized training."""
    client_datasets = {}
    all_langs = sorted(set(TRAIN_CONFIGS) | {"en"})

    for lang in all_langs:
        raw_train = load_train_split(lang)
        train_dataset = raw_train.map(
            preprocess_examples,
            batched=True,
            remove_columns=raw_train.column_names,
        )

        raw_validation = load_dataset("facebook/mlqa", EVAL_CONFIGS[lang])["validation"]
        validation_dataset = raw_validation.map(
            preprocess_examples,
            batched=True,
            remove_columns=raw_validation.column_names,
        )

        client_datasets[lang] = {
            "train": train_dataset,
            "validation": validation_dataset,
        }

    return client_datasets


def main():
    print("Preparing MLQA federated client datasets...")
    client_datasets = prepare_client_datasets()

    for lang, data in client_datasets.items():
        print(f"{lang}: train={len(data['train'])} validation={len(data['validation'])}")


if __name__ == "__main__":
    main()