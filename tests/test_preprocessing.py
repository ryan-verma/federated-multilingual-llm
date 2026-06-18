from data.preprocess_mlqa import CLIENT_LANGUAGES

print("\nFederated Clients")
print("-" * 30)

for i, lang in enumerate(CLIENT_LANGUAGES, start=1):
    print(f"Client {i}: {lang}")