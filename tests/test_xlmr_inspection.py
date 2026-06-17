from models.model import load_model, load_tokenizer

model = load_model()
tokenizer = load_tokenizer()

config = model.config

print("=" * 50)
print("XLM-R CONFIGURATION")
print("=" * 50)

print(f"Hidden Size       : {config.hidden_size}")
print(f"Hidden Layers     : {config.num_hidden_layers}")
print(f"Attention Heads   : {config.num_attention_heads}")
print(f"Intermediate Size : {config.intermediate_size}")
print(f"Max Position Emb. : {config.max_position_embeddings}")
print(f"Vocabulary Size   : {config.vocab_size}")

print("\nMODEL ARCHITECTURE")
print("-" * 50)
print(model)