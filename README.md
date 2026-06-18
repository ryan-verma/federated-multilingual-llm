# Federated Multilingual QA using XLM-R

A federated multilingual Question Answering system built using **XLM-RoBERTa** that learns from geographically distributed clients while preserving linguistic diversity and privacy.

The project investigates the trade-offs between centralized and federated training using multilingual datasets such as **MLQA**, **XNLI**, and **XTREME**.

> 🚧 **Status:** Work in Progress — this repository is being actively developed as part of an academic project.

---

## Objectives

- Develop a multilingual QA model using a pretrained transformer
- Simulate federated learning across multiple language-specific clients
- Study the effects of federated averaging on different languages
- Explore diversity-preserving aggregation methods
- Integrate privacy techniques such as Differential Privacy
- Compare centralized and federated approaches

## Technologies Used

- Python 3.12
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Flower
- Opacus
- Pandas
- Matplotlib
- Scikit-learn
- Git & GitHub
- VS Code

## Datasets

| Dataset | Description |
|---|---|
| **MLQA** | Multilingual Question Answering |
| **XNLI** | Cross-lingual Natural Language Inference |
| **XTREME** | Multilingual evaluation benchmark |

## Planned Workflow

**Phase 1 – Environment Setup**
- Configure development environment
- Install dependencies
- Set up GPU support

**Phase 2 – Model Preparation**
- Load and inspect XLM-R Base
- Analyze tokenizer behavior

**Phase 3 – Dataset Preparation**
- Download and inspect datasets
- Partition data by language

**Phase 4 – Centralized Training**
- Fine-tune the baseline multilingual QA model

**Phase 5 – Federated Learning**
- Simulate multiple Flower clients
- Implement FedAvg

**Phase 6 – Linguistic Diversity**
- Investigate language imbalance
- Implement diversity-preserving approaches

**Phase 7 – Privacy**
- Integrate Differential Privacy techniques

**Phase 8 – Evaluation**
- Compare all approaches and analyze results

## Repository Structure

```
federated-multilingual-qa/
├── data/
├── models/
├── tests/
├── requirements.txt
├── README.md
└── main.py
```

## Current Progress

- [x] Environment setup completed
- [x] GPU-enabled PyTorch configured
- [x] XLM-R Base integrated
- [x] Model inspection completed
- [x] Multilingual tokenization tested
- [x] Dataset preparation
- [ ] Centralized baseline training
- [ ] Federated learning experiments
- [ ] Diversity-preserving methods
- [ ] Privacy integration
- [ ] Final evaluation and analysis

## Team

| Member | Responsibilities |
|---|---|
| **Person 1** | Training pipeline, federated execution, experiment runs, privacy-aware training |
| **Person 2** | Dataset preparation, evaluation, analysis, diversity methods, documentation |

## Status

🚧 **Work in Progress** — this repository is being actively developed as part of an academic project.
