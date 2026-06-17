from transformers import AutoTokenizer, AutoModel
import torch


MODEL_NAME = "xlm-roberta-base"


def load_tokenizer():
    """Load XLM-R tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model():
    """Load XLM-R model."""
    return AutoModel.from_pretrained(MODEL_NAME)


def get_device():
    """Return GPU if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading model...")
    model = load_model()

    device = get_device()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print("\nModel Information")
    print("-" * 30)
    print("Model:", MODEL_NAME)
    print("Device:", device)
    print(f"Parameters: {total_params:,}")
    print("Vocabulary Size:", tokenizer.vocab_size)

    print("\nXLM-R Base loaded successfully.")