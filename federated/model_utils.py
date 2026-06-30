import torch

from transformers import (
    XLMRobertaForQuestionAnswering,
)


# Federated training now starts from the raw pretrained encoder, not the
# centralized checkpoint. This matters for your Phase 19 comparison: FedAvg,
# diversity-aware FedAvg, and DP-FedAvg all need to start from the same
# untouched point as the centralized baseline so differences in the results
# are attributable to the training PROCEDURE, not to federated training
# getting a head start from data it was never supposed to see pooled.
MODEL_NAME = "xlm-roberta-base"

# Used only for naming federated output folders (outputs/fl/<this>/<lang>)
# so runs starting from raw weights don't get mixed up with any earlier
# experiment that started from a centralized checkpoint.
FL_INIT_FROM = "raw_xlmr"


def get_model():

    model = (
        XLMRobertaForQuestionAnswering
        .from_pretrained(MODEL_NAME)
    )

    return model


def get_parameters(model):

    return [
        val.cpu().numpy()
        for _, val in model.state_dict().items()
    ]


def set_parameters(model, parameters):

    params_dict = zip(
        model.state_dict().keys(),
        parameters,
    )

    state_dict = {
        k: torch.tensor(v)
        for k, v in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True,
    )