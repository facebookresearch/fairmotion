# Motion Clustering

In this task, we semantically cluster motion sequences from a large motion capture dataset. We implement two quick methods to generate features for sequences -- the first based on joint heuristics [1] and the second based on kinetic energy [2] and acceleration of joints. We cluster the features using KMeans and Hierarchical approaches and visualize their t-SNE embeddings.

<img src="tsne-pca-k-8.jpg" width="600">

t-SNE embeddings of sequences from AMASS CMU dataset; 8 clusters formed by k-means clustering

## Instructions

Gather all motion files to be clustered in a single folder. As an optional step, we provide a tool to generate equal sized chunks of motion sequences from the dataset. `split_bvh.py` split BVH files in a folder to create overlapping `--time-window` second clips, and saves them in BVH format in the output folder.
```
python fairmotion/tasks/clustering/split_bvh.py \
    --folder $PATH_TO_BVH_DATASET \
    --output-folder $PATH_TO_STORE_SPLIT_BVH \
    --time-window 2
```
To generate feature vectors for each motion sequence from a dataset, use the `generate_features.py` script. The script offers 2 different feature sets (`manual` [1] or `kinetic` [2]). It uses multiple processes to generate features in a parallel manner.
```
python fairmotion/tasks/clustering/generate_features \
    --folder $PATH_TO_BVH_DATASET \
    --output-folder $PATH_TO_STORE_FEATURE_FILES \
    --features kinetic
```
We can cluster the feature vectors from the previous step using `clustering.py`. We use the `sklearn.cluster` module to provide several clustering techniques -- `kmeans`, `hierarchical`, `optics` and `dbscan`. We provide additional options to normalize and clip (by percentile) input features. We write results to a csv file. The results contain information about filename, cluster ID, rank within cluster and distance from cluster center.
```
python fairmotion/tasks/clustering/clustering.py \
    --features $FEATURES_FILE # see generate_features.py \
    --type kmeans \
    --num-clusters $NUM_CLUSTERS \
    --normalize-features \
    --clip-features 90 \
    --output-file $OUTPUT_CSV_FILE \
    --linkage average
```

To test code, run unit tests in `test_generate_features.py`.

## References
[1] Müller, Meinard, Tido Röder, and Michael Clausen. "Efficient content-based retrieval of motion capture data." ACM SIGGRAPH 2005

[2] Onuma, Kensuke, Christos Faloutsos, and Jessica K. Hodgins. "FMDistance: A Fast and Effective Distance Function for Motion Capture Data." Eurographics (Short Papers). 2008.