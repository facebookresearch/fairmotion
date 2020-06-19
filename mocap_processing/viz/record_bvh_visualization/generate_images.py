"""
Sample command
python mocap_processing/viz/record_bvh_visualization/generate_images.py --bvh-file ~/Downloads/scp/amass/CMU/85/85_14_poses.bvh --output-images-folder ~/85_14_images/
"""

import argparse
import numpy as np
import tqdm

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from mocap_processing.motion.pfnn import Animation, BVH
from mocap_processing.viz import gl_render, glut_viewer as viewer


def render(global_positions, frame_num, joint_parents):
    gl_render.render_ground(
        size=[100, 100], color=[0.8, 0.8, 0.8], axis="y", origin=True, use_arrow=True
    )
    glPushMatrix()
    glRotatef(90, -1, 0, 0)
    # glScalef(0.05, 0.05, 0.05)

    glEnable(GL_LIGHTING)
    for i in range(len(joint_parents)):
        pos = global_positions[frame_num][i]
        gl_render.render_point(pos, radius=0.025, color=[0.8, 0.8, 0.0, 1.0])
        j = joint_parents[i]
        if j != -1:
            pos_parent = global_positions[frame_num][j]
            gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0, 0, 1])
    glPopMatrix()


def load_animation(bvh_filename):
    animation, joint_names, time_per_frame = BVH.load(bvh_filename)
    joint_parents = animation.parents
    global_positions = Animation.positions_global(animation)
    return global_positions, joint_parents, time_per_frame, animation.positions


def keyboard_callback(key):
    global frame_num, global_positions, joint_parents, args
    if key == b"s":
        for frame_num in tqdm.tqdm(range(len(global_positions))):
            frame_num += 1
            viewer.drawGL()
            viewer.save_screen("{args.output_images_folder}", f"frame_{frame_num:03d}")
    return


def render_callback():
    global frame_num, global_positions, joint_parents

    # frame_num = calculate_current_frame(start_time, time_per_frame, global_positions)
    render(global_positions, frame_num, joint_parents)


def main(args):
    global frame_num, global_positions, joint_parents
    frame_num = 0
    global_positions, joint_parents, time_per_frame, positions = load_animation(
        args.bvh_file
    )
    cam_origin = 0.01 * np.array([0, 50, 0])
    cam_pos = cam_origin + np.array([0.0, 1.0, 3.5])
    viewer.run(
        title="BVH viewer",
        cam_pos=cam_pos,
        cam_origin=cam_origin,
        size=(1280, 720),
        keyboard_callback=keyboard_callback,
        render_callback=render_callback,
        # overlay_callback=overlay_callback,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save screen shots of BVH visualization in a folder."
        "Press s to start"
    )
    parser.add_argument("--bvh-file", type=str, required=True)
    parser.add_argument("--output-images-folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
