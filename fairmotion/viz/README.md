## Record video of BVH visualization
To save visualizations using `fairmotion/viz/bvh_visualizer.py`, follow these steps-
1. Run visualizer with appropriate input arguments
```
python fairmotion/viz/bvh_visualizer.py --bvh-files BVH_FILENAMES
```
2. Press 'r' (for record) to trigger recording feature. Enter output folder where screenshots can be saved.
3. Collate the images into a video using `ffmpeg` tool
```
ffmpeg -r 30 -f image2 -s 1600x1600 -i EMPTY_NEW_OUTPUT_FOLDER/frame_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p OUTPUT_VIDEO_NAME.mp4
```