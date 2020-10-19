import numpy as np
import torch
from fairmotion.tasks.motion_plausibility import (
    model as plausibility_model, preprocess
)
from fairmotion.utils import constants


MODEL_PATH = "/checkpoint/dgopinath/models/scadive/manifold/rotmat_5_5/0.model"
BATCH_SIZE = 2


def load_model(model_path, device="cuda"):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model = plausibility_model.MLP(
        **checkpoint["model_kwargs"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.double()
    model.eval()
    model.to(device)
    return model, checkpoint["model_kwargs"], checkpoint["stats"]


def main():
    model, model_kwargs, stats = load_model(MODEL_PATH)
    # Create random data with BATCH_SIZE
    observed_poses = np.random.rand(
        BATCH_SIZE,
        model_kwargs["num_observed"],
        int(model_kwargs["input_dim"]/3),
        3,
    )
    predicted_poses = np.random.rand(
        BATCH_SIZE, int(model_kwargs["input_dim"]/3), 3,
    )
    # To load and run the model, see following steps
    rep = "aa"
    observed_poses = preprocess.process_sample(observed_poses, rep)
    predicted_poses = preprocess.process_sample(predicted_poses, rep)

    observed_poses = torch.Tensor(
        (observed_poses - stats[0]) / (
            stats[1] + constants.EPSILON
        )
    ).double()
    predicted_poses = torch.Tensor(
        (predicted_poses - stats[0]) / (
            stats[1] + constants.EPSILON
        )
    ).double()

    score = model(observed_poses, predicted_poses)
    print(score.detach().numpy())


if __name__ == "__main__":
    main()
