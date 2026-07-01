# Federated Multilingual QA using XLM-R

A federated multilingual Question Answering system built using **XLM-RoBERTa** that learns from geographically distributed language clients while preserving linguistic diversity and data privacy.

The project investigates trade-offs between centralized and federated training across five languages using **MLQA** as the primary benchmark, comparing standard FedAvg, fairness-aware aggregation (q-FFL), and differentially private training.

> 🚧 **Status:** Work in Progress — actively developed as part of an academic research project.

---

## Objectives

- Fine-tune a multilingual QA model using XLM-RoBERTa-base
- Simulate federated learning across multiple language-specific clients
- Study the effects of federated averaging on cross-lingual transfer
- Implement diversity-preserving aggregation to address language imbalance
- Integrate Differential Privacy for privacy-aware federated training
- Compare all approaches on per-language EM/F1 metrics

---

## Technologies Used

- Python 3.10
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Flower (flwr 1.31.0)
- Opacus
- Pandas
- Matplotlib
- Scikit-learn
- Git & GitHub
- VS Code

---

## Model

**XLM-RoBERTa-base** — a 12-layer multilingual transformer pretrained on 100 languages (278M parameters), fine-tuned as an extractive QA model. A linear span-prediction head is added on top of the encoder, producing start and end logits per token.

---

## Dataset

**MLQA** (facebook/mlqa) — a multilingual extractive QA benchmark with aligned answer spans across 7 languages. Used for training and primary evaluation.

| Language | Train Source | Eval Config |
|---|---|---|
| English | SQuAD (real English QA) | mlqa.en.en |
| Hindi | mlqa-translate-train.hi | mlqa.hi.hi |
| Spanish | mlqa-translate-train.es | mlqa.es.es |
| German | mlqa-translate-train.de | mlqa.de.de |
| Chinese | mlqa-translate-train.zh | mlqa.zh.zh |

**XNLI** (xnli) — Cross-lingual Natural Language Inference benchmark. Used for broader cross-lingual transfer evaluation across all four training setups.

**XTREME** (xtreme) — Multilingual multitask benchmark covering a range of cross-lingual tasks. Used for comprehensive evaluation of cross-lingual generalization.

---

## Federated Setup

| Setting | Value |
|---|---|
| Framework | Flower (flwr) |
| Clients | 5 (one per language) |
| Rounds | 3 |
| Local epochs per round | 1 |
| Aggregation (standard) | FedAvg |
| Aggregation (weighted) | q-FFL (q=2) |
| Starting checkpoint | Raw xlm-roberta-base (all variants) |

All federated variants start from the same raw pretrained checkpoint as the centralized baseline so that performance differences are attributable to the training procedure alone, not initialization.

---

## Approaches Compared

| Approach | Description |
|---|---|
| **Centralized** | All language data pooled, standard fine-tuning |
| **FedAvg** | Standard federated averaging weighted by dataset size |
| **Weighted FedAvg** | q-FFL: clients weighted by loss^q × size, upweighting underperforming languages |
| **DP-FedAvg** | FedAvg with Differential Privacy via Opacus *(planned)* |

---

## Results

### Centralized Baseline

| Language | EM | F1 |
|---|---|---|
| English | 67.16 | 79.44 |
| Hindi | 45.36 | 64.00 |
| Spanish | 44.00 | 69.25 |
| German | 42.58 | 60.64 |
| Chinese | 18.25 | 60.46 |

> Chinese F1 uses character-level evaluation. The standard whitespace-tokenized SQuAD metric collapses F1 toward EM for Chinese (no whitespace between words), understating performance. Character-level comparison is used for zh specifically.

### Federated FedAvg — Round 1 Eval Loss

| Language | Eval Loss |
|---|---|
| English | 1.396 |
| Hindi | 1.981 |
| Spanish | 2.051 |
| German | 2.176 |
| Chinese | 2.155 |

> Final EM/F1 results pending completion of training (rounds 2 and 3 in progress).

---

## Repository Structure

```
federated-multilingual-qa/
├── data/
│   ├── preprocess_mlqa.py       # tokenization, client dataset preparation
│   └── mlqa_loader.py
├── evaluation/
│   ├── evaluate_mlqa.py         # per-language EM/F1 evaluation
│   ├── metrics.py               # span decoding (logits -> predicted text)
│   ├── preprocess_eval.py       # eval preprocessing (offset_mapping)
│   └── cjk_metrics.py          # character-level EM/F1 for Chinese
├── federated/
│   ├── client.py                # Flower NumPyClient
│   ├── client_data.py           # lazy per-language dataset loading
│   ├── server.py                # FedAvg + checkpoint saving
│   └── model_utils.py           # model loading and parameter utilities
├── federated_weighted/
│   └── server.py                # q-FFL weighted aggregation
├── models/
│   └── model.py
├── training/
│   ├── train_centralized.py     # centralized baseline entry point
│   ├── train_federated.py       # plain FedAvg entry point
│   └── train_federated_weighted.py  # q-FFL entry point
├── outputs/
│   ├── centralized/             # centralized model checkpoints
│   ├── fl/                      # FedAvg checkpoints (per round + final)
│   └── fl_weighted/             # weighted FedAvg checkpoints
├── results/
│   └── centralized/             # saved evaluation metrics (JSON)
├── requirements.txt
├── README.md
└── main.py
```

---

## Current Progress

- [x] Environment setup and GPU configuration
- [x] XLM-RoBERTa-base loaded and tokenizer tested
- [x] Multilingual tokenization pipeline with sliding window overflow
- [x] Per-language client dataset preparation (train + eval splits)
- [x] Centralized baseline training (3 epochs, fp16)
- [x] Per-language EM/F1 evaluation with character-level fix for Chinese
- [x] Flower federated setup with 5 language clients
- [x] Standard FedAvg implementation with per-round checkpoint saving
- [x] Resume support for interrupted federated runs
- [x] q-FFL language-weighted FedAvg implemented
- [ ] FedAvg training complete (round 1 done, rounds 2-3 in progress)
- [ ] Weighted FedAvg training and evaluation
- [ ] Differential Privacy integration (Opacus)
- [ ] XNLI and XTREME evaluation across all four setups
- [ ] Per-language analysis and plots
- [ ] Final four-way comparison
