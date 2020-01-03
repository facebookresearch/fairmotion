from sklearn.manifold import TSNE

import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_tsne_embeddings(X):
    X = np.array(X)
    X_embedded = TSNE(
        n_components=2, init='random', random_state=0,
    ).fit_transform(X)
    return X_embedded


def plot_tsne_embeddings(filename, X, labels=None):
    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(X[:,0], X[:,1], marker=",")
    else:
        for l in np.unique(labels):
            i = np.where(labels == l)
            ax.scatter(X[i,0], X[i,1], label=l, marker=",")
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
    tsne_embeddings = get_tsne_embeddings(all_features)
    filename2label = {}
    if args.clusters_file:
        with open(args.clusters_file) as f:
            for line in f:
                label_and_score, filename = line.split(":")
                label = int(label_and_score.strip().split(",")[0])
                filename2label[filename.strip()] = label
    plot_tsne_embeddings(
        filename=args.output_file, 
        X=tsne_embeddings,
        labels=[
            filename2label[filename] for filename in filenames
        ] if args.clusters_file else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate images with t-sne visualization'
    )
    parser.add_argument("--features-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--clusters-file", type=str)
    args = parser.parse_args()
    main(args)