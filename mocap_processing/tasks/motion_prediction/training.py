import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from mocap_processing.tasks.motion_prediction import generate, utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


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
    logging.info(args._get_kwargs())
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device else device
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(args.preprocessed_path, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        args.batch_size,
        device,
    )

    # number of predictions per time step = num_joints * angle representation
    data_shape = next(iter(dataset["train"]))[0].shape
    num_predictions = data_shape[-1]

    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )
    utils.create_dir_if_absent(args.save_model_path)

    criterion = nn.MSELoss()
    model.apply(init_weights)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=5
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    training_losses, val_losses = [], []

    logging.info("Training model...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        teacher_forcing_ratio = np.clip(
            (1 - 2*epoch/args.epochs),
            a_min=0,
            a_max=1,
        )
        logging.info(
            f"Running epoch {epoch} | "
            f"teacher_forcing_ratio={teacher_forcing_ratio}"
        )
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            optimizer.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(
                src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
            )
            outputs = outputs.double()
            loss = criterion(outputs, tgt_seqs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / ((iterations + 1) * args.batch_size)
        training_losses.append(epoch_loss)
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
        val_losses.append(val_loss)
        logging.info(
            f"Training loss {epoch_loss} | "
            f"Validation loss {val_loss} | "
            f"Iterations {iterations + 1}"
        )
        if scheduler:
            scheduler.step(val_loss)
        if epoch % args.save_model_frequency == 0:
            torch.save(
                model.state_dict(),
                f"{args.save_model_path}/{epoch}.model"
            )
            if len(val_losses) == 0 or val_loss <= min(val_losses):
                torch.save(
                    model.state_dict(),
                    f"{args.save_model_path}/best.model"
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
        "--preprocessed-path", type=str, help="Path to folder with pickled "
        "files", required=True
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
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
    parser.add_argument(
        "--device", type=str, help="Training device", default=None,
        choices=["cpu", "cuda"]
    )
    parser.add_argument(
        "--architecture", type=str, help="Seq2Seq archtiecture to be used",
        default="seq2seq",
        choices=[
            "seq2seq", "tied_seq2seq", "transformer", "transformer_encoder",
        ]
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=0.5,
    )
    parser.add_argument(
        "--optimizer", type=str, help="Torch optimizer",
        default="sgd", choices=["adam", "sgd"]
    )
    args = parser.parse_args()
    main(args)
