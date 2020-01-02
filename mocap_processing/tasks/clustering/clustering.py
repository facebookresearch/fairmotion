"""
Example command
python mocap_processing/tasks/clustering/clustering.py --output-file ~/data/clustering_random_pfnn/random_pfnn_translated_kinetic_local_position_features/clusters.tsv --features ~/data/clustering_random_pfnn/random_pfnn_translated_kinetic_local_position_features/features.tsv --num-clusters 10
"""

import argparse
import numpy as np
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, OPTICS, cluster_optics_dbscan
from collections import defaultdict


def calculate_score(centroid, features):
    return np.linalg.norm(features - centroid)


def get_ranked_clusters(clusters):
    """
    Input:
    clusters: defaultdict where items are in the format cluster_idx: [(name, score)]

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


def run_optics_clustering(features, names, args):
    optics = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)


def run_kmeans_clustering(features, names, args):
    kmeans = KMeans(args.num_clusters).fit(features)
    clusters = defaultdict(list)
    for num, label in enumerate(kmeans.labels_):
        score = calculate_score(kmeans.cluster_centers_[label], features[num])
        clusters[label].append((names[num], score))

    ranked_clusters = get_ranked_clusters(clusters)
    
    with open(args.output_file, "w") as f:
        for cluster in ranked_clusters:
            for (name, rank, score) in ranked_clusters[cluster]:
                f.write(",".join([str(cluster), str(rank), str(score)]) + ":" + str(name) + "\n")


def run_hierarchical_clustering(features, names, args):
    hierarchical = AgglomerativeClustering(args.num_clusters, linkage=args.linkage).fit(features)
    cluster_centroids = defaultdict(lambda: np.zeros(len(features[0])))
    for num, label in enumerate(hierarchical.labels_):
        cluster_centroids[label] += np.array(features[num])
    # Average sum of points to get centroids
    for cluster in cluster_centroids:
        cluster_centroids[cluster] = cluster_centroids[cluster]/len(cluster_centroids[cluster])

    clusters = defaultdict(list)
    for num, label in enumerate(hierarchical.labels_):
        clusters[label].append((names[num], calculate_score(cluster_centroids[label], features[num])))
    
    ranked_clusters = get_ranked_clusters(clusters)
    
    with open(args.output_file, "w") as f:
        for cluster in ranked_clusters:
            for (name, rank, score) in ranked_clusters[cluster]:
                f.write(",".join([str(cluster), str(rank), str(score)]) + ":" + str(name) + "\n")

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
            features.append([float(x) for x in line.split(":")[-1].split("\t")])
    if 0.0 < args.clip_features < 100.0:
        np.percentile(features, args.clip_features, axis=0, overwrite_input=True)
    if args.normalize_features:
        features = normalize_features(features)
    if args.type == "kmeans":
        run_kmeans_clustering(features, names, args)
    elif args.type == "hierarchical":
        run_hierarchical_clustering(features, names, args)
    elif args.type == "optics":
        run_optics_clustering(features, names, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster features with kMeans")
    parser.add_argument("--features", type=str, help="Features tsv file")
    parser.add_argument(
        "--output-file", type=str, help="File to store information about clusters"
    )
    parser.add_argument(
        "--type", type=str, required=True,
        choices=["kmeans", "hierarchical"],
        help="Clustering technique to be used, one of kmeans and hierarchical", 
    )
    parser.add_argument(
        "--num-clusters", type=int, help="Number of clusters", required=True,
    )
    parser.add_argument(
        "--linkage", type=str, help="Type of linkage in agglomerative clusering",
        default="average"
    )
    parser.add_argument(
        "--normalize-features", action='store_true', 
        help="Perform feature normalization",
    )
    parser.add_argument(
        "--clip-features", type=float, help="Clip feature by percentile",
        default=95
    )
    args = parser.parse_args()
    main(args)
