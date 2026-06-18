CLIENT_LANGUAGES = {
    "en": "mlqa.en.en",
    "hi": "mlqa.hi.hi",
    "es": "mlqa.es.es",
    "de": "mlqa.de.de",
    "fr": "mlqa.fr.fr",
}

from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def load_language_dataset(config):
    return load_dataset("facebook/mlqa", config)

sample = dataset["validation"][0]

print(sample.keys())

def preprocess_examples(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )