# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn

from fairmotion.models import optimizer
from fairmotion.tasks.motion_manifold import data, model as manifold_model
from fairmotion.utils import utils as fairmotion_utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_model(input_dim, hidden_dim, num_layers, num_observed, device):
    model = manifold_model.MLP(
        input_dim=input_dim,
        num_observed=num_observed,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        device=device,
    )
    model = model.to(device)
    model.zero_grad()
    model.double()
    return model


def prepare_optimizer(opt_type, model, lr):
    opt = None
    if optimizer == "adam":
        opt = optimizer.AdamOpt(model, lr=lr)
    else:
        opt = optimizer.SGDOpt(model, lr=lr)
    return opt


def train(args):
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    logging.info(args._get_kwargs())
    fairmotion_utils.log_config(args.save_model_path, args)

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device else device
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    dataset = data.get_loader(
        args.preprocessed_file,
        args.batch_size,
        device,
    )

    batch_size, num_observed, input_dim = next(iter(dataset))[0].shape

    model = prepare_model(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_observed=num_observed,
        device=device,
    )

    criterion = nn.MSELoss()
    model.init_weights()

    epoch_loss = 0

    logging.info("Training model...")
    opt = prepare_optimizer(args.optimizer, model, args.lr)
    training_losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        for iterations, (observed, predicted, label) in enumerate(dataset):
            opt.optimizer.zero_grad()
            output = model(observed, predicted)
            output = output.double()
            loss = criterion(
                output,
                label,
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / (iterations + 1)
        training_losses.append(epoch_loss)
        logging.info(
            f"Training loss {epoch_loss} | "
            f"Iterations {iterations + 1}"
        )
    torch.save(model.state_dict(), f"{args.save_model_path}/{epoch}.model")
    return training_losses


def plot_curves(args, training_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")


def main(args):
    train_losses = train(args)
    plot_curves(args, train_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constrastive motion manifold training"
    )
    parser.add_argument(
        "--preprocessed-file",
        type=str,
        help="Path to pickled file with preprocessed data",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=256,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=2,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=10
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer to use",
        default="adam",
        choices=["adam", "sgd"],
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=0.001,
    )
    args = parser.parse_args()
    main(args)
