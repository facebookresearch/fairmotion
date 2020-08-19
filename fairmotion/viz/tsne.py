# Copyright (c) Facebook, Inc. and its affiliates.

from sklearn.manifold import MDS, TSNE

import argparse
import matplotlib.pyplot as plt
import numpy as np


def normalize_features(features):
    X = np.array(features)
    Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
    return (X - Xmean) / (Xstd + 1.0e-8)


def get_tsne_embeddings(X):
    X = np.array(X)
    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X)
    return X_embedded


def get_mds_embeddings(X):
    X = np.array(X)
    X_embedded = MDS(n_components=2, random_state=0,).fit_transform(X)
    return X_embedded


def plot_embeddings(filename, X, labels=None):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], marker="o")
    else:
        # In order of count of labels, highest to lowest
        for l in np.flip(np.argsort(np.bincount(labels))):
            i = np.where(labels == l)
            ax.scatter(X[i, 0], X[i, 1], label=l, marker="o")
    ax.legend()
    plt.savefig(filename)


def main(args):
    all_features = []
    filenames = []
    with open(args.features_file) as f:
        for line in f:
            filename, features = line.split(":")
            all_features.append(np.array(list(map(float, features.strip().split()))))
            filenames.append(filename.strip())
    norm_features = normalize_features(all_features)
    if args.algorithm == "tsne":
        embeddings = get_tsne_embeddings(norm_features)
    elif args.algorithm == "mds":
        embeddings = get_mds_embeddings(norm_features)
    filename2label = {}
    if args.clusters_file:
        with open(args.clusters_file) as f:
            for line in f:
                label_and_score, filename = line.split(":")
                label = int(label_and_score.strip().split(",")[0])
                filename2label[filename.strip()] = label
    plot_embeddings(
        filename=args.output_file,
        X=embeddings,
        labels=[filename2label[filename] for filename in filenames]
        if args.clusters_file
        else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images with t-sne visualization"
    )
    parser.add_argument("--features-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--clusters-file", type=str)
    parser.add_argument("--algorithm", choices=["mds", "tsne"], required=True)
    args = parser.parse_args()
    main(args)
