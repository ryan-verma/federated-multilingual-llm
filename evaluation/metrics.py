import collections
import numpy as np


def compute_metrics(
    start_logits,
    end_logits,
    features,
    examples,
):

    example_to_features = collections.defaultdict(list)

    for idx, feature in enumerate(features):
        example_to_features[
            feature["example_id"]
        ].append(idx)

    predictions = {}

    for example in examples:

        example_id = example["id"]

        context = example["context"]

        feature_indices = example_to_features[
            example_id
        ]

        best_answer = {
            "text": "",
            "score": -1e9,
        }

        for feature_index in feature_indices:

            start_logit = start_logits[
                feature_index
            ]

            end_logit = end_logits[
                feature_index
            ]

            offsets = features[
                feature_index
            ]["offset_mapping"]

            start_indexes = np.argsort(
                start_logit
            )[-20:]

            end_indexes = np.argsort(
                end_logit
            )[-20:]

            for start_index in start_indexes:

                for end_index in end_indexes:

                    if (
                        offsets[start_index] is None
                        or offsets[end_index] is None
                    ):
                        continue

                    if end_index < start_index:
                        continue

                    if end_index - start_index + 1 > 30:
                        continue

                    start_char = offsets[
                        start_index
                    ][0]

                    end_char = offsets[
                        end_index
                    ][1]

                    text = context[
                        start_char:end_char
                    ]

                    score = (
                        start_logit[start_index]
                        + end_logit[end_index]
                    )

                    if score > best_answer["score"]:

                        best_answer = {
                            "text": text,
                            "score": score,
                        }

        predictions[example_id] = best_answer[
            "text"
        ]

    return predictions