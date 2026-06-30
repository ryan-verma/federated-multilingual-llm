from flwr.app import Context
from flwr.server import (
    ServerAppComponents,
    ServerConfig,
)

from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp

"""
Minimal Flower server configuration for the federated learning setup.

num_rounds=3 chosen to roughly match the centralized baseline's total
training exposure per language: each client trains 1 local epoch per
round, so 3 rounds = 3 full local epochs per language overall, comparable
to the centralized script's 3 epochs over the pooled dataset. (Not a
strict equivalent -- federated training resets each client's weights to
the global average at the start of every round, so each "epoch" picks up
from a different starting point than continuous centralized training
does. That difference is part of what the comparison is meant to surface.)
"""

def server_fn(context: Context):

    strategy = FedAvg()

    config = ServerConfig(
        num_rounds=3
    )

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


server_app = ServerApp(
    server_fn=server_fn
)