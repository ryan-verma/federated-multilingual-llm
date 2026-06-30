import copy
import os

import flwr as fl

from federated.client_data import load_client_dataset

from federated import model_utils
from federated.model_utils import (
    get_model,
    get_parameters,
    set_parameters,
)

import torch

from transformers import (
    Trainer,
    TrainingArguments,
)

# Caches the loaded model state_dict per process so repeated client
# instantiations (e.g. across rounds, if Flower recreates the client object)
# don't re-read the checkpoint file from disk each time -- it gets
# overwritten by set_parameters() on the very first fit() call anyway, so
# disk weights are only ever truly needed once per process.
_MODEL_STATE_CACHE = {}


def _get_cached_model():
    from transformers import XLMRobertaForQuestionAnswering

    if "state_dict" not in _MODEL_STATE_CACHE:
        # First call in this process: real disk read.
        model = get_model()
        _MODEL_STATE_CACHE["config"] = model.config
        _MODEL_STATE_CACHE["state_dict"] = copy.deepcopy(model.state_dict())
        return model

    # Subsequent calls: build the architecture from the cached config only
    # (no weight file read), then load the cached in-memory weights.
    model = XLMRobertaForQuestionAnswering(_MODEL_STATE_CACHE["config"])
    model.load_state_dict(_MODEL_STATE_CACHE["state_dict"])
    return model

class MLQAClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.language, self.dataset = (
            load_client_dataset(client_id)
        )

        # Debug subset: only applied if FL_DEBUG_SUBSET env var is set, so a
        # real run can never silently inherit this. Example:
        #   FL_DEBUG_SUBSET=500 python -m federated.simulation
        debug_subset_size = os.environ.get("FL_DEBUG_SUBSET")
        if debug_subset_size is not None:
            n = min(int(debug_subset_size), len(self.dataset["train"]))
            self.dataset["train"] = self.dataset["train"].select(range(n))
            print(f"[DEBUG] {self.language} train subset: {n} examples")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = _get_cached_model()
        self.model.to(self.device)

        print(
            f"Client {self.language} initialized "
            f"on {self.device}"
        )

    def get_parameters(self, config):
            return get_parameters(
                self.model
            )

    def fit(self, parameters, config):
        set_parameters(
            self.model,
            parameters,
        )

        training_args = TrainingArguments(
        output_dir=f"outputs/fl/{model_utils.FL_INIT_FROM}/{self.language}",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=3e-5,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
        )

        print(
            f"Training client: {self.language}"
        )

        print(
            "Training examples:",
            len(self.dataset["train"])
        )

        before = self.model.qa_outputs.weight.detach().cpu().clone()

        trainer.train()

        after = self.model.qa_outputs.weight.detach().cpu()

        difference = torch.mean(
            torch.abs(after - before)
        )

        print(
            f"{self.language} mean weight change:",
            difference.item()
        )

        updated_parameters = get_parameters(
            self.model
        )

        return (
            updated_parameters,
            len(self.dataset["train"]),
            {},
        )

    def evaluate(
        self,
        parameters,
        config,
        ):

        set_parameters(
            self.model,
            parameters,
        )

        # Real eval: run a forward pass over the validation set and return
        # the actual loss. Trainer.evaluate() needs args even when we're not
        # training, mainly for per_device_eval_batch_size and a writable
        # output_dir; everything else can stay default.
        eval_args = TrainingArguments(
            output_dir=f"outputs/fl/{model_utils.FL_INIT_FROM}/{self.language}/eval",
            per_device_eval_batch_size=4,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.dataset["validation"],
        )

        metrics = trainer.evaluate()
        loss = metrics["eval_loss"]

        print(
            f"{self.language} eval loss:",
            loss
        )

        return (
            loss,
            len(self.dataset["validation"]),
            {"eval_loss": loss},
        )


def client_fn(context):

    client_id = context.node_config["partition-id"]

    return MLQAClient(
        int(client_id)
    ).to_client()


client_app = fl.clientapp.ClientApp(
    client_fn=client_fn
)