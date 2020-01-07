# Motion Library

The Motion Library provides easy-to-use interfaces and tools to work with motion capture data. The objective of the library is to hide the complexity of mocap file formats, 3D geometry, representation and visualization, and let users focus on high level tasks with motion.

<img src="mocap_processing/viz/samples/anim_viz.gif" width="500"><img src="mocap_processing/viz/samples/anim_smpl.gif" width="330">

## Installation
To run a clean install of the library, first create a virtual environment with Python3, and activate it.
```
virtualenv --python=python3 FOLDER/mocap_processing
. FOLDER/mocap_processing/bin/activate
```
Now install the project using `pip`. This will also pull in external dependencies [basecode](https://github.com/Jungdam/basecode/) and [amass](https://github.com/nghorbani/amass).
```
pip install -e .
```
