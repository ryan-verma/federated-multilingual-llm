from datasets import load_dataset

dataset = load_dataset("xnli", "all_languages")

print("\nXNLI Splits")
print("-" * 30)

for split in dataset:
    print(split, len(dataset[split]))

print("\nSample Example")
print("-" * 30)

sample = dataset["train"][0]

print("Available Languages:")
print(list(sample["premise"].keys()))

print("\nEnglish Premise:")
print(sample["premise"]["en"])

print("\nEnglish Hypothesis:")
idx = sample["hypothesis"]["language"].index("en")
print(sample["hypothesis"]["translation"][idx])

print("\nLabel:", sample["label"])