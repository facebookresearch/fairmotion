from sklearn.cluster import KMeans
from collections import defaultdict

def main():
    features = []
    names = []
    with open("/private/home/dgopinath/data/pfnn_locomation_bvh_split_features/features.tsv") as f:
        for line in f:
            features.append([int(x) for x in line.strip().split("\t")])
    with open("/private/home/dgopinath/data/pfnn_locomation_bvh_split_features/labels.tsv") as f:
        for line in f:
            names.append(line.strip())
    kmeans = KMeans(20).fit(features)
    clusters = defaultdict(list)
    print(kmeans.labels_)
    print(len(features))
    for num, label in enumerate(kmeans.labels_):
        clusters[label].append(names[num])
    with open("/private/home/dgopinath/data/pfnn_locomation_bvh_split_features/clusters.tsv", "w") as f:
        for cluster in clusters:
            for name in clusters[cluster]:
                f.write(f"{cluster}:{name}\n")

if __name__ == "__main__":
    main()
