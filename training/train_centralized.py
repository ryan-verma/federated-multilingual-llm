import os
import torch

from datasets import concatenate_datasets
from transformers import (
    AutoTokenizer,
    XLMRobertaForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

from data.preprocess_mlqa import prepare_client_datasets


def main():

    # ==================================================
    # Environment Information
    # ==================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nEnvironment")
    print("-" * 30)
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ==================================================
    # Load QA Model
    # ==================================================
    model = XLMRobertaForQuestionAnswering.from_pretrained("xlm-roberta-base")
    model.to(device)

    print("\nModel")
    print("-" * 30)
    print("Loaded: xlm-roberta-base")
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # ==================================================
    # Load Federated Client Datasets
    # ==================================================
    client_datasets = prepare_client_datasets()

    # ==================================================
    # Create Centralized Datasets
    # ==================================================
    train_dataset = concatenate_datasets(
        [client_datasets[lang]["train"] for lang in client_datasets]
    )

    validation_dataset = concatenate_datasets(
        [client_datasets[lang]["validation"] for lang in client_datasets]
    )

    print("\nDatasets")
    print("-" * 30)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    # ==================================================
    # Configure Training Arguments
    # ==================================================
    training_args = TrainingArguments(
        output_dir="outputs/centralized",
        logging_dir="logs",
        eval_strategy="epoch",
        save_strategy="steps",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_steps=10000,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    # ==================================================
    # Initialize Trainer
    # ==================================================
    # Plain Trainer is enough here -- this eval_dataset has start/end labels
    # (from preprocess_examples), so Trainer can compute a normal eval loss
    # during training. Real per-language EM/F1 is computed separately,
    # post-training, by evaluation/evaluate_mlqa.py.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    print("\nTrainer initialized successfully.")

    # Only resume if a checkpoint actually exists -- a first run has nothing
    # to resume from and `resume_from_checkpoint=True` would error.
    checkpoint_dir = training_args.output_dir
    has_checkpoint = os.path.isdir(checkpoint_dir) and any(
        name.startswith("checkpoint-") for name in os.listdir(checkpoint_dir)
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=has_checkpoint)

    print("\nSaving model...")
    trainer.save_model("outputs/centralized/final_model")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    tokenizer.save_pretrained("outputs/centralized/final_model")

    print("\nTraining complete.")
    print("Run evaluation/evaluate_mlqa.py separately for per-language EM/F1.")


if __name__ == "__main__":
    main()