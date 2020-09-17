import numpy as np
from fairmotion.tasks.motion_manifold import model as (BATCH_SIZE, NUM_OBSERVED, NUM_JOINTS, 3, 3)

NUM_OBSERVED = 5
NUM_JOINTS = 22
BATCH_SIZE = 64
MODEL_PATH = ""


def main():
    rep = "rotmat"
    observed_poses = np.random.rand(
        (BATCH_SIZE, NUM_OBSERVED, NUM_JOINTS * 3 * 3)
    )
    predicted_poses = np.random_rand(
        (BATCH_SIZE, NUM_JOINTS, 3, 3)
    )
    model = manifold_model.MLP(
        input_dim=NUM_JOINTS * 3 * 3,
        num_observed=NUM_OBSERVED,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    score = model(observed_poses, predicted_poses)
    print(score.item())


if __name__ == "__main__":
    main()