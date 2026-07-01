from data.preprocess_mlqa import load_single_client_dataset


LANGUAGES = ["en", "hi", "es", "de", "zh"]


def load_client_dataset(client_id):
    """Loads only the language this specific client needs, instead of
    building all five upfront -- each Flower client process only ever uses
    one language, so there's no reason to pay the preprocessing cost for
    the other four."""
    language = LANGUAGES[client_id]
    return language, load_single_client_dataset(language)