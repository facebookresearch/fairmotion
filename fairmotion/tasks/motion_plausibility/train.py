# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn

from fairmotion.models import optimizer
from fairmotion.tasks.motion_plausibility import (
    data, model as plausibility_model, options
)
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
    model = plausibility_model.MLP(
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


def prepare_criterion(loss_fn):
    if loss_fn == "ce":
        return nn.BCELoss()
    else:
        return nn.MSELoss()


def save_model(model, epoch, save_model_path, model_kwargs, stats, metadata, save_as_best=False):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "stats": stats,
            "metadata": metadata
        },
        f"{save_model_path}/{epoch}.model",
    )
    if save_as_best:
        torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "stats": stats,
            "metadata": metadata
        },
        f"{save_model_path}/best.model",
    )


def train(
    save_model_path,
    train_preprocessed_file,
    valid_preprocessed_file,
    hidden_dim,
    num_layers,
    optimizer,
    lr,
    epochs,
    batch_size=256,
    device=None,
):
    fairmotion_utils.create_dir_if_absent(save_model_path)

    set_seeds()
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if not device else device
    )
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    train_dataloader = data.get_loader(
        train_preprocessed_file,
        batch_size,
        device,
    )
    valid_dataloader = data.get_loader(
        valid_preprocessed_file,
        batch_size,
        device,
    )

    batch_size, num_observed, input_dim = next(iter(train_dataloader))[0].shape

    model_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_observed": num_observed,
        "device": device,
    }
    model = prepare_model(
        **model_kwargs
    )
    criterion = prepare_criterion(args.criterion)
    model.init_weights()

    epoch_loss = 0
    prev_valid_loss = float("inf")

    logging.info("Training model...")
    opt = prepare_optimizer(optimizer, model, lr)
    training_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for iterations, (observed, predicted, label) in enumerate(
            train_dataloader
        ):
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

        valid_loss = 0
        model.eval()
        for iterations, (observed, predicted, label) in enumerate(
            valid_dataloader
        ):
            output = model(observed, predicted)
            output = output.double()
            loss = criterion(
                output,
                label,
            )
            valid_loss += loss.item()
        valid_loss = valid_loss / (iterations + 1)
        logging.info(
            f"Epoch: {epoch} | "
            f"Training loss: {epoch_loss} | "
            f"Validation loss: {valid_loss} | "
            f"Iterations: {iterations + 1}"
        )
        if epoch % 10 == 0:
            save_as_best = False
            if valid_loss < prev_valid_loss:
                save_as_best = True
                prev_valid_loss = valid_loss
            save_model(
                model=model,
                epoch=epoch,
                save_model_path=save_model_path,
                model_kwargs=model_kwargs,
                stats=[
                    train_dataloader.dataset.mean, train_dataloader.dataset.std
                ],
                metadata=train_dataloader.dataset.metadata,
                save_as_best=save_as_best,
            )

    save_model(
        model=model,
        epoch=epoch,
        save_model_path=save_model_path,
        model_kwargs=model_kwargs,
        stats=[train_dataloader.dataset.mean, train_dataloader.dataset.std],
        metadata=train_dataloader.dataset.metadata,
    )
    return training_losses


def plot_curves(save_model_path, training_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{save_model_path}/loss.svg", format="svg")


def main(args):
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    logging.info(args._get_kwargs())
    fairmotion_utils.log_config(args.save_model_path, args)
    train_losses = train(
        args.save_model_path,
        args.train_preprocessed_file,
        args.valid_preprocessed_file,
        args.hidden_dim,
        args.num_layers,
        args.optimizer,
        args.lr,
        args.epochs,
        args.batch_size,
        args.device,
    )
    plot_curves(args.save_model_path, train_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constrastive motion plausibility model training"
    )
    parser = options.add_train_args(parser)
    args = parser.parse_args()
    main(args)
