from datasets import load_dataset


def load_mlqa():
    """
    Load MLQA dataset from Hugging Face.
    """
    dataset = load_dataset("mlqa")

    return dataset


if __name__ == "__main__":
    dataset = load_mlqa()

    print(dataset)