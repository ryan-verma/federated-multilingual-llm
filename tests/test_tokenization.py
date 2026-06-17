from models.model import load_tokenizer

tokenizer = load_tokenizer()

samples = {
    "English": "What is your name?",
    "Hindi": "आपका नाम क्या है?",
    "French": "Comment vous appelez-vous ?",
    "Spanish": "¿Cómo te llamas?",
    "German": "Wie heißt du?"
}

for language, text in samples.items():
    encoding = tokenizer(text)

    print("\n" + "=" * 40)
    print(language)
    print(text)
    print("Token IDs:", encoding["input_ids"])
    print("Number of tokens:", len(encoding["input_ids"]))