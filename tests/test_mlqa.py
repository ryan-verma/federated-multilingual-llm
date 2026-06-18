from datasets import get_dataset_config_names

configs = get_dataset_config_names("facebook/mlqa")

print("Available MLQA Configurations:")
print("-" * 40)

for config in configs:
    print(config)

print("\nTotal:", len(configs))