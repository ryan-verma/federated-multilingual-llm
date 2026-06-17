import sys
import torch
import transformers
import datasets
import flwr
import opacus
import evaluate
import pandas
import matplotlib
import sklearn

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Transformers Version: {transformers.__version__}")
print(f"Datasets Version: {datasets.__version__}")
print(f"Flower Version: {flwr.__version__}")
print(f"Opacus Version: {opacus.__version__}")
print(f"Pandas Version: {pandas.__version__}")
print(f"Matplotlib Version: {matplotlib.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")

print("\nAll imports successful.")
print("Environment setup complete.")