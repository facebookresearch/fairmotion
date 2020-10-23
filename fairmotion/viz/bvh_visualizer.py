# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import os
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh, asfamc
from fairmotion.ops import conversions, math, motion as motion_ops
from fairmotion.utils import utils


class MocapViewer(glut_viewer.Viewer):
    """
    MocapViewer is an extension of the glut_viewer.Viewer class that implements
    requisite callback functions -- render_callback, keyboard_callback,
    idle_callback and overlay_callback.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1
    ```

    To visualize more than 1 motion sequence side by side, append more files 
    to the `--bvh-files` argument. Set `--x-offset` to an appropriate float 
    value to add space separation between characters in the row.

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --bvh-files $BVH_FILE1 $BVH_FILE2 $BVH_FILE3 \
        --x-offset 2
    ```

    To visualize asfamc motion sequence:

    ```
    python fairmotion/viz/bvh_visualizer.py \
        --asf-files tests/data/11.asf \
        --amc-files tests/data/11_01.amc
    ```

    """

    def __init__(
        self,
        motions,
        play_speed=1.0,
        scale=1.0,
        thickness=1.0,
        render_overlay=False,
        hide_origin=False,
        **kwargs,
    ):
        self.motions = motions
        self.play_speed = play_speed
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.file_idx = 0
        self.cur_time = 0.0
        self.scale = scale
        self.thickness = thickness
        super().__init__(**kwargs)

    def keyboard_callback(self, key):
        motion = self.motions[self.file_idx]
        if key == b"s":
            self.cur_time = 0.0
            self.time_checker.begin()
        elif key == b"]":
            next_frame = min(
                motion.num_frames() - 1,
                motion.time_to_frame(self.cur_time) + 1,
            )
            self.cur_time = motion.frame_to_time(next_frame)
        elif key == b"[":
            prev_frame = max(0, motion.time_to_frame(self.cur_time) - 1)
            self.cur_time = motion.frame_to_time(prev_frame)
        elif key == b"+":
            self.play_speed = min(self.play_speed + 0.2, 5.0)
        elif key == b"-":
            self.play_speed = max(self.play_speed - 0.2, 0.2)
        elif (key == b"r" or key == b"v"):
            self.cur_time = 0.0
            end_time = motion.length()
            fps = motion.fps
            save_path = input(
                "Enter directory/file to store screenshots/video: "
            )
            cnt_screenshot = 0
            dt = 1 / fps
            gif_images = []
            while self.cur_time <= end_time:
                print(
                    f"Recording progress: {self.cur_time:.2f}s/{end_time:.2f}s ({int(100*self.cur_time/end_time)}%) \r",
                    end="",
                )
                if key == b"r":
                    utils.create_dir_if_absent(save_path)
                    name = "screenshot_%04d" % (cnt_screenshot)
                    self.save_screen(dir=save_path, name=name, render=True)
                else:
                    image = self.get_screen(render=True)
                    gif_images.append(
                        image.convert("P", palette=Image.ADAPTIVE)
                    )
                self.cur_time += dt
                cnt_screenshot += 1
            if key == b"v":
                utils.create_dir_if_absent(os.path.dirname(save_path))
                gif_images[0].save(
                    save_path,
                    save_all=True,
                    optimize=False,
                    append_images=gif_images[1:],
                    loop=0,
                )
        else:
            return False

        return True

    def _render_pose(self, pose, body_model, color):
        skel = pose.skel
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            gl_render.render_point(pos, radius=0.03 * self.scale, color=color)
            if j.parent_joint is not None:
                # returns X that X dot vec1 = vec2
                pos_parent = conversions.T2p(
                    pose.get_transform(j.parent_joint, local=False)
                )
                p = 0.5 * (pos_parent + pos)
                l = np.linalg.norm(pos_parent - pos)
                r = 0.1 * self.thickness
                R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
                gl_render.render_capsule(
                    conversions.Rp2T(R, p),
                    l,
                    r * self.scale,
                    color=color,
                    slice=8,
                )

    def _render_characters(self, colors):
        for i, motion in enumerate(self.motions):
            t = self.cur_time % motion.length()
            skel = motion.skel
            pose = motion.get_pose_by_frame(motion.time_to_frame(t))
            color = colors[i % len(colors)]

            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            glEnable(GL_LIGHTING)
            self._render_pose(pose, "stick_figure2", color)

    def render_callback(self):
        gl_render.render_ground(
            size=[100, 100],
            color=[0.8, 0.8, 0.8],
            axis=utils.axis_to_str(self.motions[0].skel.v_up_env),
            origin=not self.hide_origin,
            use_arrow=True,
        )
        colors = [
            np.array([123, 174, 85, 255]) / 255.0,  # green
            np.array([255, 255, 0, 255]) / 255.0,  # yellow
            np.array([85, 160, 173, 255]) / 255.0,  # blue
        ]
        self._render_characters(colors)

    def idle_callback(self):
        time_elapsed = self.time_checker.get_time(restart=False)
        self.cur_time += self.play_speed * time_elapsed
        self.time_checker.begin()

    def overlay_callback(self):
        if self.render_overlay:
            w, h = self.window_size
            t = self.cur_time % self.motions[0].length()
            frame = self.motions[0].time_to_frame(t)
            gl_render.render_text(
                f"Frame number: {frame}",
                pos=[0.05 * w, 0.95 * h],
                font=GLUT_BITMAP_TIMES_ROMAN_24,
            )


def main(args):
    v_up_env = utils.str_to_axis(args.axis_up)
    if args.bvh_files:
        motions = [
            bvh.load(
                file=filename,
                v_up_skel=v_up_env,
                v_face_skel=utils.str_to_axis(args.axis_face),
                v_up_env=v_up_env,
                scale=args.scale,
            )
            for filename in args.bvh_files
        ]
    else:
        motions = [
            asfamc.load(file=f, motion=m)
            for f, m in zip(args.asf_files, args.amc_files)
        ]

    for i in range(len(motions)):
        motion_ops.translate(motions[i], [args.x_offset * i, 0, 0])
    cam = camera.Camera(
        pos=np.array(args.camera_position),
        origin=np.array(args.camera_origin),
        vup=v_up_env,
        fov=45.0,
    )
    viewer = MocapViewer(
        motions=motions,
        play_speed=args.speed,
        scale=args.scale,
        thickness=args.thickness,
        render_overlay=args.render_overlay,
        hide_origin=args.hide_origin,
        title="Motion Graph Viewer",
        cam=cam,
        size=(1280, 720),
    )
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize BVH file with block body"
    )
    parser.add_argument("--bvh-files", type=str, nargs="+", required=False)
    parser.add_argument("--asf-files", type=str, nargs="+", required=False)
    parser.add_argument("--amc-files", type=str, nargs="+", required=False)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--thickness", type=float, default=1.0,
        help="Thickness (radius) of character body"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--axis-up", type=str, choices=["x", "y", "z"], default="z"
    )
    parser.add_argument(
        "--axis-face", type=str, choices=["x", "y", "z"], default="y"
    )
    parser.add_argument(
        "--camera-position",
        nargs="+",
        type=float,
        required=False,
        default=(2.0, 2.0, 2.0),
    )
    parser.add_argument(
        "--camera-origin",
        nargs="+",
        type=float,
        required=False,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--hide-origin", action="store_true")
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument(
        "--x-offset",
        type=int,
        default=2,
        help="Translates each character by x-offset*idx to display them "
        "simultaneously side-by-side",
    )
    args = parser.parse_args()
    assert len(args.camera_position) == 3 and len(args.camera_origin) == 3, (
        "Provide x, y and z coordinates for camera position/origin like "
        "--camera-position x y z"
    )
    main(args)
