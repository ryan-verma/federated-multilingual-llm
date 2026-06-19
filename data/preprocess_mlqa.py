from datasets import load_dataset
from transformers import AutoTokenizer

TRAIN_CONFIGS = {
    "en": "mlqa-translate-train.es",
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

MODEL_NAME = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_language_dataset(config):
    return load_dataset("facebook/mlqa", config)

def preprocess_examples(examples):
    questions = [q.strip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):

        answer = examples["answers"][i]

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1

        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if (
            offsets[context_start][0] > start_char
            or offsets[context_end][1] < end_char
        ):
            start_positions.append(0)
            end_positions.append(0)

        else:
            token_start = context_start

            while (
                token_start < len(offsets)
                and offsets[token_start][0] <= start_char
            ):
                token_start += 1

            start_positions.append(token_start - 1)

            token_end = context_end

            while (
                token_end >= 0
                and offsets[token_end][1] >= end_char
            ):
                token_end -= 1

            end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized

def prepare_client_datasets():

    client_datasets = {}

    for lang in TRAIN_CONFIGS:

        train_dataset = load_language_dataset(
            TRAIN_CONFIGS[lang]
        )

        train_dataset = train_dataset.map(
            preprocess_examples,
            batched=True,
            remove_columns=train_dataset["train"].column_names,
        )

        eval_dataset = load_language_dataset(
            EVAL_CONFIGS[lang]
        )

        eval_dataset = eval_dataset.map(
            preprocess_examples,
            batched=True,
            remove_columns=eval_dataset["validation"].column_names,
        )

        client_datasets[lang] = {
            "train": train_dataset["train"],
            "validation": eval_dataset["validation"],
            "test": eval_dataset["test"],
        }

    return client_datasets

def main():
    print("Preparing MLQA client datasets...")

    client_datasets = {}

    for lang, config in EVAL_CONFIGS.items():

        print(f"\nProcessing {lang}...")

        dataset = load_language_dataset(config)

        tokenized_dataset = dataset.map(
            preprocess_examples,
            batched=True,
            remove_columns=dataset["validation"].column_names,
        )

        client_datasets[lang] = tokenized_dataset

        print(
            f"{lang} validation examples:",
            len(tokenized_dataset["validation"])
        )

    sample = client_datasets["en"]["validation"][0]

    print("\nVerification")
    print("-" * 30)

    print("Input length:",
        len(sample["input_ids"]))

    print("Start position:",
        sample["start_positions"])

    print("End position:",
        sample["end_positions"])

if __name__ == "__main__":
    main()
