import torch

from datasets import concatenate_datasets
from transformers import (
    XLMRobertaForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

from data.preprocess_mlqa import prepare_client_datasets

def main():

    # ==================================================
    # Environment Information
    # ==================================================
    # Detect device
    # Print device and GPU information

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nEnvironment")
    print("-" * 30)
    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ==================================================
    # Load QA Model
    # ==================================================
    # Load XLM-R QA model
    # Move model to device
    # Print model information

    model = XLMRobertaForQuestionAnswering.from_pretrained(
        "xlm-roberta-base"
    )

    model.to(device)

    print("\nModel")
    print("-" * 30)
    print("Loaded: xlm-roberta-base")
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    # ==================================================
    # Load Federated Client Datasets
    # ==================================================
    # Load preprocessed MLQA datasets
    # (en, hi, es, de, zh)

    client_datasets = prepare_client_datasets()

    # ==================================================
    # Create Centralized Datasets
    # ==================================================
    # Combine client train datasets
    # Combine client validation datasets
    # Print dataset statistics

    train_dataset = concatenate_datasets(
        [client_datasets[lang]["train"]
        for lang in client_datasets]
    )

    validation_dataset = concatenate_datasets(
        [client_datasets[lang]["validation"]
        for lang in client_datasets]
    )

    print("\nDatasets")
    print("-" * 30)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    # ==================================================
    # Configure Training Arguments
    # ==================================================
    # Initialize TrainingArguments

    training_args = TrainingArguments(
        output_dir="results",
        logging_dir="logs",
    )

    # ==================================================
    # Initialize Trainer
    # ==================================================
    # Create Hugging Face Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # ==================================================
    # Verification
    # ==================================================
    # Print success message

    print("\nTrainer initialized successfully.")

if __name__ == "__main__":
    main()