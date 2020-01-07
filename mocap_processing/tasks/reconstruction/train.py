import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict
from itertools import product
from sklearn import model_selection

from mocap_processing.processing.pfnn import preprocessing
from mocap_processing.motion.pfnn import BVH
from mocap_processing.models import seq2seq
from mocap_processing.tasks.reconstruction import generate


def get_dataset_files(dataset_path):
    angles = [45, 90, 135, 180]
    frequencies = [30, 60, 90, 120]
    types = ["sinusoidal", "sinusoidal_x"] #"pendulum"

    dataset_files = []
    for angle, freq, motion_type in product(angles, frequencies, types):
        filename = os.path.join(dataset_path, motion_type, "source", f"{angle}_{freq}.bvh")
        dataset_files.append(filename)
    return dataset_files


def prepare_dataset(dataset_files):
    dataset = defaultdict(list)
    for filename in dataset_files:
        anim, _, frametime = BVH.load(filename)
        joint_rotations_root, joint_positions_root = preprocessing.positions_wrt_base(anim)
        _, input_data, _, _ = preprocessing.extract_sequences(joint_rotations_root[1], joint_positions_root, frametime)
        # input_data is of shape [num_sequences, seq_len, num_predictions]
        train, test = model_selection.train_test_split(input_data, train_size=0.8)
        dataset["train"].extend(train)
        dataset["test"].extend(test)
    return dataset


def prepare_model(input_dim, hidden_dim, device):
    enc = seq2seq.LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    dec = seq2seq.LSTMDecoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = seq2seq.Seq2Seq(enc, dec).to(device)
    model.zero_grad()
    model.double()
    return model


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    set_seeds()

    dataset_files = get_dataset_files(args.dataset_path)
    dataset = prepare_dataset(dataset_files)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # input_dim is num_predictions, which is num_joints*2*3 because there are 
    # 3 angles and 3 position values for each joint
    num_predictions = next(iter(dataset["train"])).shape[1]

    model = prepare_model(input_dim=num_predictions, hidden_dim=args.hidden_dim, device=device)

    criterion = nn.MSELoss()    
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)
    training_losses, val_losses = [], []
    
    print(f"""Training dataset size {len(dataset["train"])}""")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        sequences = dataset["train"]
        epoch_loss = 0
        model.train()
        for i in range(int(len(sequences)/args.batch_size)):
            batched_data_np = np.array(sequences[i*args.batch_size:(i+1)*args.batch_size])
            batched_data_t = torch.from_numpy(batched_data_np).to(device=device)
            batched_target_data_t = batched_data_t.transpose(0, 1)
            optimizer.zero_grad()
            outputs = model(batched_data_t, batched_target_data_t)
            outputs = outputs.to(dtype=torch.double)
            loss = criterion(outputs, batched_data_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/(len(sequences)/args.batch_size)
        training_losses.append(epoch_loss)
        print(f"Training loss {epoch_loss}")
        
        val_loss = generate.eval(model, criterion, dataset["test"])
        val_losses.append(val_loss)
        print(f"Validation loss {val_loss}")
        
        scheduler.step(val_loss)
        if epoch % args.save_model_frequency == 0:
            torch.save(model.state_dict(), f"{args.save_model_path}/{epoch}.model")
    return training_losses, val_losses


def plot_curves(args, training_losses, val_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")


def main(args):
    train_losses, val_losses = train(args)
    plot_curves(args, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence training with motion files"
    )
    parser.add_argument(
        "--dataset-path", type=str, help="Path to BVH motion files", 
        required=True
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training",
        default=64
    )
    parser.add_argument(
        "--hidden-dim", type=int, help="Hidden size of LSTM units in encoder/decoder",
        default=128
    )
    parser.add_argument(
        "--save-model-path", type=str, help="Path to store saved models", 
        required=True
    )
    parser.add_argument(
        "--save-model-frequency", type=int, 
        help="Frequency (in terms of number of epochs) at which model is saved",
        default=5
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs",
        default=200
    )
    args = parser.parse_args()
    main(args)