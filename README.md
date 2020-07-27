# Motion Library

The Motion Library provides easy-to-use interfaces and tools to work with motion capture data. The objective of the library is to hide the complexity of mocap file formats, 3D geometry, motion representation and visualization, and let users focus on high level learning tasks.

<img src="mocap_processing/viz/samples/anim_viz.gif" width="500"><img src="mocap_processing/viz/samples/anim_smpl.gif" width="330">

## Installation
First, clone the repository.
```
git clone https://github.com/fairinternal/mocap_processing.git
```
Before installing the project, we recommend creating a virtual environment with Python3, and activating it.
```
virtualenv --python=python3 $VENV_FOLDER/mocap_processing
. $VENV_FOLDER/mocap_processing/bin/activate
```
Now install the project using `pip`. This will also pull in external dependency [amass](https://github.com/nghorbani/amass).
```
pip install -e .
```

## Tasks
The `tasks` module showcases practical usages of the motion classes, models and visualization tools. Below, we list tasks that have been used in different projects. They build the basic infrastructure to enable incremental addition of more features. 

### Clustering of motion capture dataset
In this task, we semantically cluster motion sequences from a large motion capture dataset, specifically the [AMASS dataset](http://amass.is.tue.mpg.de/). We implement two quick methods to generate features for sequences -- the first based on [joint heuristics](https://dl.acm.org/doi/10.1145/1073204.1073247) and the second based on [kinetic energy](https://www.researchgate.net/publication/251419971_FMDistance_A_fast_and_effective_distance_function_for_motion_capture_data) and acceleration of joints. We cluster the features using KMeans and Hierarchical approaches and visualize their t-SNE embeddings.

<img src="mocap_processing/tasks/clustering/tsne-pca-k-8.jpg" width="600">

t-SNE embeddings of sequences from AMASS CMU dataset; 8 clusters formed by k-means clustering

### Changepoint detection
We implement an acceleration based changepoint detection algorithm in `mocap_processing/tasks/changepoint_detection`.

![changepoints](mocap_processing/tasks/changepoint_detection/changepoints.svg)

## License
Mephisto is MIT licensed. See the LICENSE file for details.