import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from mocap_processing.models import seq2seq
from mocap_processing.tasks.motion_prediction import dataset as motion_dataset
from mocap_processing.tasks.motion_prediction import generate


def prepare_dataset(train_path, valid_path, test_path, batch_size):
    dataset = {}
    for split, split_path in zip(
        ["train", "test", "valid"],
        [train_path, valid_path, test_path]
    ):
        dataset[split] = motion_dataset.get_loader(split_path, batch_size)
    return dataset


def prepare_model(input_dim, hidden_dim, device):
    enc = seq2seq.LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
    dec = seq2seq.LSTMDecoder(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim
    )
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

    dataset = prepare_dataset(
        *[
            os.path.join(args.processed_data, f"{split}.pkl")
            for split in ["train", "test", "valid"]
        ],
        args.batch_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # number of predictions per time step = num_joints * angle representation
    data_shape = next(iter(dataset["train"])).shape
    num_predictions = data_shape[-2] * data_shape[-1]

    model = prepare_model(
        input_dim=num_predictions, hidden_dim=args.hidden_dim, device=device
    )

    criterion = nn.MSELoss()
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=5
    )
    training_losses, val_losses = [], []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0
        model.train()
        iterations = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            optimizer.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(src_seqs, tgt_seqs)
            outputs = outputs.to(dtype=torch.double)
            loss = criterion(outputs, tgt_seqs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (iterations * args.batch_size)
        training_losses.append(epoch_loss)
        print(f"Training loss {epoch_loss}")

        val_loss = generate.eval(
            model, criterion, dataset["valid"], args.batch_size, device,
        )
        val_losses.append(val_loss)
        print(f"Validation loss {val_loss}")

        scheduler.step(val_loss)
        if epoch % args.save_model_frequency == 0:
            torch.save(
                model.state_dict(),
                f"{args.save_model_path}/{epoch}.model"
            )
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
        description="Sequence to sequence motion prediction training"
    )
    parser.add_argument(
        "--processed-path", type=str, help="Path to folder containing pickled "
        "files", required=True
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=128,
    )
    parser.add_argument(
        "--save-model-path", type=str, help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is "
        "saved",
        default=5,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=200
    )
    args = parser.parse_args()
    main(args)
