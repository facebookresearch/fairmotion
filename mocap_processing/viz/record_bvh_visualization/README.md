## Record video of BVH visualization
To save BVH visualization from `mocap_processing/viz/Mocap_Visualizer` notebook, follow these steps-
1. Save screenshots of each frame of BVH visualization by running
```
python mocap_processing/viz/record_bvh_visualization/generate_images.py --bvh-file BVH_FILENAME --output-images-folder EMPTY_NEW_OUTPUT_FOLDER
```
2. Collate the images into a video using `ffmpeg` tool
```
ffmpeg -r 30 -f image2 -s 1600x1600 -i EMPTY_NEW_OUTPUT_FOLDER/frame_%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p OUTPUT_VIDEO_NAME.mp4
```