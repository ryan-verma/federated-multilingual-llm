from datasets import load_dataset

dataset = load_dataset(
    "xnli",
    "all_languages"
)

print("\nXNLI Splits")
print("-" * 30)

for split in dataset:
    print(split, len(dataset[split]))

print("\nSample Example")
print("-" * 30)

sample = dataset["train"][0]

print("Premise:", sample["premise"])
print("Hypothesis:", sample["hypothesis"])
print("Label:", sample["label"])
print("Language:", sample["language"])