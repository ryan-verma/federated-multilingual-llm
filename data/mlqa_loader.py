from datasets import load_dataset

DATASET_NAME = "facebook/mlqa"


def load_mlqa(language_pair="mlqa.en.en"):
    """
    Load an MLQA configuration.
    Example: mlqa.en.en
    """
    return load_dataset(DATASET_NAME, language_pair)


if __name__ == "__main__":
    dataset = load_mlqa()

    print("\nDataset Splits")
    print("-" * 30)

    for split in dataset:
        print(split, len(dataset[split]))

    print("\nSample Example")
    print("-" * 30)

    sample = dataset["test"][0]

    print("ID:", sample["id"])
    print("Context:", sample["context"][:300], "...")
    print("Question:", sample["question"])
    print("Answers:", sample["answers"])