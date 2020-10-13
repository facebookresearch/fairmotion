# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import numpy as np
import os
import torch
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

from fairmotion.viz import camera, gl_render, glut_viewer
from fairmotion.data import bvh
from fairmotion.ops import conversions, math
from fairmotion.tasks.motion_plausibility import test
from fairmotion.utils import constants, utils


class MotionManifoldViewer(glut_viewer.Viewer):
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
    """

    def __init__(
        self,
        motion,
        model_path,
        render_overlay=False,
        hide_origin=False,
        **kwargs,
    ):
        self.motion = motion
        self.render_overlay = render_overlay
        self.hide_origin = hide_origin
        self.file_idx = 0
        self.cur_time = 0.0
        self.score = 0

        self.model, self.model_kwargs, self.model_stats = test.load_model(
            model_path,
        )
        super().__init__(**kwargs)

    def keyboard_callback(self, key):
        motion = self.motion
        if key == b"s":
            self.cur_time = 0.0
            self.time_checker.begin()
        elif (key == b"r" or key == b"v"):
            self.cur_time = 0.0
            end_time = self.motion.length()
            fps = self.motion.fps
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
                    os.makedirs(save_path, exist_ok=True)
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

    def _render_pose(self, pose, observed):
        if observed is None:
            color = np.array([0., 0., 255., 255.]) / 255.0
            self.score = 0
        else:
            observed = torch.Tensor(
                (
                    utils.flatten_angles(conversions.R2A(observed), "aa") -
                    self.model_stats[0]
                )/(
                    self.model_stats[1] + constants.EPSILON
                )
            ).double()
            pose_t = torch.Tensor(
                (
                    utils.flatten_angles(
                        conversions.R2A(pose.rotations()), "aa"
                    ) - self.model_stats[0]
                )/(
                    self.model_stats[1] + constants.EPSILON
                )
            ).double()
            self.score = self.model(observed, pose_t).data.cpu().numpy()[0][0]
            color = np.array(
                [(1-self.score)*255, self.score*255, 0, 255]
            ) / 255.0
        skel = pose.skel
        for j in skel.joints:
            T = pose.get_transform(j, local=False)
            pos = conversions.T2p(T)
            gl_render.render_point(pos, radius=0.03, color=color)
            if j.parent_joint is not None:
                # returns X that X dot vec1 = vec2
                pos_parent = conversions.T2p(
                    pose.get_transform(j.parent_joint, local=False)
                )
                p = 0.5 * (pos_parent + pos)
                l = np.linalg.norm(pos_parent - pos)
                r = 0.05
                R = math.R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
                gl_render.render_capsule(
                    conversions.Rp2T(R, p),
                    l,
                    r,
                    color=color,
                    slice=8,
                )

    def _render_characters(self):
        t = self.cur_time % self.motion.length()
        frame = self.motion.time_to_frame(t)
        pose = self.motion.get_pose_by_frame(frame)

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_LIGHTING)

        num_observed = self.model_kwargs["num_observed"]
        skip_frames = 2
        num_observed * skip_frames
        observed = None
        if frame >= num_observed * skip_frames:
            observed_frames = np.arange(
                frame - skip_frames,
                frame - (num_observed + 1) * skip_frames,
                -skip_frames,
            )
            observed = self.motion.rotations()[observed_frames]

        self._render_pose(pose, observed)

    def render_callback(self):
        gl_render.render_ground(
            size=[100, 100],
            color=[0.8, 0.8, 0.8],
            axis=utils.axis_to_str(self.motion.skel.v_up_env),
            origin=not self.hide_origin,
            use_arrow=True,
        )
        self._render_characters()

    def idle_callback(self):
        time_elapsed = self.time_checker.get_time(restart=False)
        self.cur_time += time_elapsed
        self.time_checker.begin()

    def overlay_callback(self):
        if self.render_overlay:
            w, h = self.window_size
            t = self.cur_time % self.motion.length()
            frame = self.motion.time_to_frame(t)
            gl_render.render_text(
                f"Frame number: {frame} | Score: {self.score}",
                pos=[0.05 * w, 0.95 * h],
                font=GLUT_BITMAP_TIMES_ROMAN_24,
            )


def main(args):
    v_up_env = utils.str_to_axis(args.axis_up)
    motion = bvh.load(
        file=args.bvh_file,
        v_up_skel=v_up_env,
        v_face_skel=utils.str_to_axis(args.axis_face),
        v_up_env=v_up_env,
    )
    cam = camera.Camera(
        pos=4 * utils.str_to_axis(args.axis_face)
        + 2 * utils.str_to_axis(args.axis_up),
        origin=np.array([0, 0, 0]),
        vup=v_up_env,
        fov=45.0,
    )
    viewer = MotionManifoldViewer(
        motion=motion,
        model_path=args.model_path,
        render_overlay=args.render_overlay,
        hide_origin=args.hide_origin,
        title="Motion Manifold Viewer",
        cam=cam,
        size=(1280, 720),
    )
    viewer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize performance of motion model"
    )
    parser.add_argument("--bvh-file", type=str, required=True)
    parser.add_argument(
        "--axis-up", type=str, choices=["x", "y", "z"], default="z"
    )
    parser.add_argument(
        "--axis-face", type=str, choices=["x", "y", "z"], default="y"
    )
    parser.add_argument("--hide-origin", action="store_true")
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
