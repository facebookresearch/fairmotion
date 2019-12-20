import argparse
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import defaultdict


def calculate_score(centroid, features):
    return np.linalg.norm(features - centroid)


def main(args):
    features = []
    names = []
    with open(args.features) as f:
        for line in f:
            line = line.strip()
            names.append(line.split(":")[0])
            features.append([float(x) for x in line.split(":")[-1].split("\t")])        
    kmeans = KMeans(20).fit(features)
    clusters = defaultdict(list)
    for num, label in enumerate(kmeans.labels_):
        score = calculate_score(kmeans.cluster_centers_[label], features[num])
        clusters[label].append((names[num], score))

    ranked_clusters = defaultdict(list)
    for label in clusters:
        sorted_cluster = sorted(clusters[label], key=lambda entry: entry[1])
        ranked_cluster = []
        for rank, (name, score) in enumerate(sorted_cluster):
            ranked_cluster.append((name, rank, score))
        ranked_clusters[label] = ranked_cluster    
    
    with open(args.output_file, "w") as f:
        for cluster in ranked_clusters:
            for (name, rank, score) in ranked_clusters[cluster]:
                f.write(",".join([str(cluster), str(rank), str(score)]) + ":" + str(name) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster features with kMeans")
    parser.add_argument("--features", type=str, help="Features tsv file")
    parser.add_argument(
        "--output-file", type=str, help="File to store information about clusters"
    )
    parser.add_argument(
        "--num-clusters", type=int, help="Number of clusters", required=True,
    )
    args = parser.parse_args()
    main(args)
