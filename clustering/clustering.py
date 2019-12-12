import argparse
import os
from sklearn.cluster import KMeans
from collections import defaultdict


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
    print(kmeans.labels_)
    print(len(features))
    for num, label in enumerate(kmeans.labels_):
        clusters[label].append(names[num])
    with open(os.path.join(args.output_folder, "clusters.tsv"), "w",) as f:
        for cluster in clusters:
            for name in clusters[cluster]:
                f.write(str(cluster) + ":" + str(name) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster features with kMeans")
    parser.add_argument("--features", type=str, help="Features tsv file")
    parser.add_argument(
        "--output-folder", type=str, help="Folder to output clusters.csv"
    )
    args = parser.parse_args()
    main(args)
