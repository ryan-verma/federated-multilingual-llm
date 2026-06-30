import os

from flwr.app import Context
from flwr.server import (
    ServerAppComponents,
    ServerConfig,
)

from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp

from federated import model_utils

"""
Language-weighted FedAvg server configuration.

This is a SKELETON -- aggregation weighting logic is not implemented yet.
Reuses federated/client.py, federated/client_data.py, and
federated/model_utils.py unchanged, since the weighting change only
affects how the SERVER combines client updates, not what each client does
locally.

Checkpoint/final-model saving is already wired in (same approach as
federated/server.py) so results from this variant land in their own
output folder and don't overwrite plain FedAvg's results.
"""

NUM_ROUNDS = 3
OUTPUT_ROOT = f"outputs/fl_weighted/{model_utils.FL_INIT_FROM}/global"


class LanguageWeightedFedAvg(FedAvg):
    """TODO: override aggregate_fit (or configure_fit) to change how client
    updates are weighted, instead of the default proportional-to-dataset-
    size weighting FedAvg uses. Decide between:
      - simple inverse-size correction (smaller clients get relatively
        more weight than their raw example count alone would give them)
      - q-FFL-style loss-based weighting (clients with currently higher
        loss get more weight, dynamically, each round)
    For now this behaves identically to plain FedAvg aside from saving
    checkpoints to a different output folder.
    """

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self._save_checkpoint(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def _save_checkpoint(self, parameters, server_round):
        ndarrays = parameters_to_ndarrays(parameters)

        model = model_utils.get_model()
        model_utils.set_parameters(model, ndarrays)

        is_final_round = server_round == NUM_ROUNDS
        save_dir = (
            f"{OUTPUT_ROOT}/final_model"
            if is_final_round
            else f"{OUTPUT_ROOT}/checkpoint-round-{server_round}"
        )

        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_NAME)
        tokenizer.save_pretrained(save_dir)

        print(f"[server] Saved global model checkpoint to {save_dir}")


def server_fn(context: Context):

    strategy = LanguageWeightedFedAvg()

    config = ServerConfig(
        num_rounds=NUM_ROUNDS
    )

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


server_app = ServerApp(
    server_fn=server_fn
)