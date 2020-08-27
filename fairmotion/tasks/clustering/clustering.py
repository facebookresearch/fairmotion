# Copyright (c) Facebook, Inc. and its affiliates.

"""
Example command
python fairmotion/tasks/clustering/clustering.py \
    --features $FEATURES_FILE # see generate_features.py \
    --type kmeans \
    --num-clusters $NUM_CLUSTERS \
    --normalize-features \
    --clip-features 90 \
    --output-file $OUTPUT_CSV_FILE \
    --linkage average
"""

import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, OPTICS
from collections import defaultdict


def calculate_score(centroid, features):
    return np.linalg.norm(features - centroid)


def get_ranked_clusters(clusters):
    """
    Input:
    clusters: defaultdict where items are in the format
    cluster_idx: [(name, score)]

    Output:
    ranked_clusters: defaultdict where key is cluster index, and entry is
    ordered list of (name, rank, score) tuples
    """
    ranked_clusters = defaultdict(list)
    for label in clusters:
        sorted_cluster = sorted(clusters[label], key=lambda entry: entry[1])
        ranked_cluster = []
        for rank, (name, score) in enumerate(sorted_cluster):
            ranked_cluster.append((name, rank, score))
        ranked_clusters[label] = ranked_cluster

    return ranked_clusters


def calculate_cluster_centroids(features, labels):
    cluster_centroids = defaultdict(lambda: np.zeros(len(features[0])))
    for num, label in enumerate(labels):
        cluster_centroids[label] += np.array(features[num])
    # Average sum of points to get centroids
    for cluster in cluster_centroids:
        cluster_centroids[cluster] = cluster_centroids[cluster] / len(
            cluster_centroids[cluster]
        )
    return cluster_centroids


def run_dbscan_clustering(features, names, args):
    dbscan = DBSCAN(eps=3, min_samples=2).fit(features)
    cluster_centroids = calculate_cluster_centroids(features, dbscan.labels_)
    clusters = defaultdict(list)
    for num, label in enumerate(dbscan.labels_):
        clusters[label].append(
            (
                names[num],
                calculate_score(cluster_centroids[label], features[num]),
            )
        )
    return clusters


def run_optics_clustering(features, names, args):
    optics = OPTICS(min_samples=10).fit(features)
    cluster_centroids = calculate_cluster_centroids(features, optics.labels_)
    clusters = defaultdict(list)
    for num, label in enumerate(optics.labels_):
        clusters[label].append(
            (
                names[num],
                calculate_score(cluster_centroids[label], features[num]),
            )
        )
    return clusters


def run_kmeans_clustering(features, names, args):
    kmeans = KMeans(args.num_clusters).fit(features)
    clusters = defaultdict(list)
    for num, label in enumerate(kmeans.labels_):
        score = calculate_score(kmeans.cluster_centers_[label], features[num])
        clusters[label].append((names[num], score))

    return clusters


def run_hierarchical_clustering(features, names, args):
    hierarchical = AgglomerativeClustering(
        args.num_clusters, linkage=args.linkage
    ).fit(features)
    cluster_centroids = calculate_cluster_centroids(
        features, hierarchical.labels_
    )
    clusters = defaultdict(list)
    for num, label in enumerate(hierarchical.labels_):
        clusters[label].append(
            (
                names[num],
                calculate_score(cluster_centroids[label], features[num]),
            )
        )

    return clusters


def normalize_features(features):
    X = np.array(features)
    Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
    return (X - Xmean) / (Xstd + 1.0e-8)


def main(args):
    features = []
    names = []
    with open(args.features) as f:
        for line in f:
            line = line.strip()
            names.append(line.split(":")[0])
            features.append(
                [float(x) for x in line.split(":")[-1].split("\t")]
            )

    if 0.0 < args.clip_features < 100.0:
        np.percentile(
            features, args.clip_features, axis=0, overwrite_input=True
        )
    if args.normalize_features:
        features = normalize_features(features)

    if args.type == "kmeans":
        clusters = run_kmeans_clustering(features, names, args)
    elif args.type == "hierarchical":
        clusters = run_hierarchical_clustering(features, names, args)
    elif args.type == "optics":
        clusters = run_optics_clustering(features, names, args)
    elif args.type == "dbscan":
        clusters = run_dbscan_clustering(features, names, args)

    ranked_clusters = get_ranked_clusters(clusters)

    with open(args.output_file, "w") as f:
        for cluster in ranked_clusters:
            for (name, rank, score) in ranked_clusters[cluster]:
                f.write(
                    ",".join([str(cluster), str(rank), str(score)])
                    + ":"
                    + str(name)
                    + "\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster features with kMeans"
    )
    parser.add_argument("--features", type=str, help="Features tsv file")
    parser.add_argument(
        "--output-csv",
        type=str,
        help="File to store information about clusters",
        required=True,
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["kmeans", "hierarchical", "optics", "dbscan"],
        help="Clustering technique to be used, one of kmeans and hierarchical",
    )
    parser.add_argument(
        "--num-clusters", type=int, help="Number of clusters", required=True,
    )
    parser.add_argument(
        "--linkage",
        type=str,
        help="Type of linkage in agglomerative clusering. See documentation in"
        " scikit-learn https://scikit-learn.org/stable/modules/generated/"
        " sklearn.cluster.AgglomerativeClustering.html",
        choices=["ward", "complete", "average", "single"],
        default="ward",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Perform feature normalization",
    )
    parser.add_argument(
        "--clip-features",
        type=float,
        help="Clip feature by percentile",
        default=95,
    )
    args = parser.parse_args()
    main(args)
