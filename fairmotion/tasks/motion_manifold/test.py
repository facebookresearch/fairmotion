import numpy as np
import torch
from fairmotion.tasks.motion_manifold import (
    data as manifold_data, model as manifold_model, preprocess
)
from fairmotion.utils import constants


MODEL_PATH = "/checkpoint/dgopinath/models/scadive/manifold/rotmat_5_5/0.model"
# You will find these configs in MODEL_PATH/config.txt
NUM_OBSERVED = 5
NUM_JOINTS = 22
BATCH_SIZE = 2
PKL_DATA_PATH = "/checkpoint/dgopinath/data/scadive/manifold/rotmat/data_num_observed_5_skip_frames_5.pkl"


def main():
    # Create random data
    observed_poses = np.random.rand(
        BATCH_SIZE, NUM_OBSERVED, NUM_JOINTS, 3, 3
    )
    predicted_poses = np.random.rand(
        BATCH_SIZE, NUM_JOINTS, 3, 3
    )

    # To load and run the model, see following steps
    rep = "rotmat"
    observed_poses = preprocess.process_sample(observed_poses, rep)
    predicted_poses = preprocess.process_sample(predicted_poses, rep)
    dataset = manifold_data.Dataset(PKL_DATA_PATH, "cpu")

    observed_poses = torch.Tensor(
        (observed_poses - dataset.mean) / (
            dataset.std + constants.EPSILON
        )
    ).double()
    predicted_poses = torch.Tensor(
        (predicted_poses - dataset.mean) / (
            dataset.std + constants.EPSILON
        )
    ).double()

    model = manifold_model.MLP(
        input_dim=NUM_JOINTS * 3 * 3,
        num_observed=NUM_OBSERVED,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.double()
    model.eval()

    score = model(observed_poses, predicted_poses)
    print(score.detach().numpy())


if __name__ == "__main__":
    main()