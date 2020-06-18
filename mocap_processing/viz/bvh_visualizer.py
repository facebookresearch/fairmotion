import argparse
from functools import partial
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from mocap_processing.viz import (
    camera, gl_render, glut_viewer as viewer, utils as viz_utils
)
from mocap_processing.data import bvh
from mocap_processing.processing import operations
from mocap_processing.utils import conversions, utils


def keyboard_callback(state, key):
    cur_time = state["cur_time"]
    time_checker_auto_play = state["time_checker_auto_play"]
    time_checker_global = state["time_checker_global"]
    motions = state["motions"]
    play_speed = state["play_speed"]
    args = state["args"]
    file_idx = state["file_idx"]

    motion = motions[file_idx]

    if key == b"r":
        cur_time = 0.0
        time_checker_auto_play.begin()
        time_checker_global.begin()
    elif key == b"]":
        pose1 = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
        vel1 = motion.get_velocity_by_frame(motion.time_to_frame(cur_time))
        next_frame = min(motion.num_frame() - 1, motion.time_to_frame(cur_time) + 1)
        cur_time = motion.frame_to_time(next_frame)
        pose2 = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
        vel2 = motion.get_velocity_by_frame(motion.time_to_frame(cur_time))
    elif key == b"[":
        prev_frame = max(0, motion.time_to_frame(cur_time) - 1)
        cur_time = motion.frame_to_time(prev_frame)
    elif key == b"+":
        play_speed = min(play_speed + 0.2, 5.0)
    elif key == b"-":
        play_speed = max(play_speed - 0.2, 0.2)
    elif key == b"h":
        pose = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
        heights = []
        for j in pose.skel.joints:
            p = conversions.T2p(pose.get_transform(j, local=False))
            p = operations.projectionOnVector(p, pose.skel.v_up_env)
            heights.append(np.linalg.norm(p))
        print(np.min(heights), np.max(heights))
    elif key == b",":
        file_idx_prev = file_idx
        file_idx = max(0, file_idx - 1)
        if file_idx_prev != file_idx:
            cur_time = 0.0
    elif key == b".":
        file_idx_prev = file_idx
        file_idx = min(len(motions) - 1, file_idx + 1)
        if file_idx_prev != file_idx:
            cur_time = 0.0
    elif key == b"c":
        start_time = 0.0
        end_time = input("Enter end time (s): ")
        fps = input("Enter fps (Hz): ")
        try:
            end_time = float(end_time)
            fps = float(fps)
        except ValueError:
            print("That is not a number!")
            return

        save_dir = input("Enter subdirectory for screenshot file: ")

        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError:
            print("Invalid Subdirectory")
            return

        cnt_screenshot = 0
        time_processed = start_time
        state["cur_time"] = start_time
        dt = 1 / fps
        while state["cur_time"] <= end_time:
            name = "screenshot_%04d" % (cnt_screenshot)
            p = conversions.T2p(motion.get_pose_by_time(cur_time).get_root_transform())
            viewer.drawGL()
            viewer.save_screen(dir=save_dir, name=name)
            print("\rtime_elased:", state["cur_time"], "(", name, ")", end=" ")
            state["cur_time"] += dt
            cnt_screenshot += 1
    else:
        return False

    state["cur_time"] = cur_time
    state["time_checker_auto_play"] = time_checker_auto_play
    state["time_checker_global"] = time_checker_global
    state["motions"] = motions
    state["play_speed"] = play_speed
    state["args"] = args
    state["file_idx"] = file_idx

    return True


def render_pose(pose, body_model, color, scale=1.0):
    skel = pose.skel
    for j in skel.joints:
        T = pose.get_transform(j, local=False)
        pos = conversions.T2p(T)
        gl_render.render_point(pos, radius=0.03 * scale, color=color)
        if j.parent_joint is not None:
            # returns X that X dot vec1 = vec2
            pos_parent = conversions.T2p(pose.get_transform(j.parent_joint, local=False))
            p = 0.5 * (pos_parent + pos)
            l = np.linalg.norm(pos_parent - pos)
            r = 0.05
            R = operations.get_R_from_vectors(np.array([0, 0, 1]), pos_parent - pos)
            gl_render.render_capsule(conversions.Rp2T(R, p), l, r * scale, color=color, slice=8)


def render_characters(motions, cur_time, colors, scale=1.0):
    for i, motion in enumerate(motions):
        time_offset = 0.0
        t = (cur_time + time_offset) % motion.length()
        skel = motion.skel
        pose = motion.get_pose_by_frame(motion.time_to_frame(t))
        color = colors[i % len(colors)]

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_LIGHTING)
        render_pose(pose, "stick_figure2", color, scale)


def render_callback(state):
    cur_time = state["cur_time"]
    motions = state["motions"]
    v_up_env = state["v_up_env"]
    hide_origin = state["args"].hide_origin
    scale = state["args"].scale

    gl_render.render_ground(
        size=[100, 100], color=[0.8, 0.8, 0.8], axis=utils.axis_to_str(v_up_env), origin=not hide_origin, use_arrow=True
    )
    colors = [
        np.array([123, 174, 85, 255]) / 255.0,  # green
        np.array([220, 220, 220, 120]) / 255.0,  # grey
        np.array([85, 160, 173, 255]) / 255.0,  # blue
    ]
    render_characters(motions, cur_time, colors, scale)


def idle_callback(state):
    cur_time = state["cur_time"]
    time_checker_auto_play = state["time_checker_auto_play"]
    play_speed = state["play_speed"]

    time_elapsed = time_checker_auto_play.get_time(restart=False)
    cur_time += play_speed * time_elapsed
    time_checker_auto_play.begin()

    state.update({
        "cur_time": cur_time,
        "time_checker_auto_play": time_checker_auto_play,
    })


def overlay_callback(state):
    if state["args"].render_overlay:
        w, h = viewer.window_size
        # print(state["cur_time"])
        # print(state["motions"][0].length())
        t = state["cur_time"] % state["motions"][0].length()
        frame = state['motions'][0].time_to_frame(t)
        status = "SEED SEQUENCE" if frame < 120 else "PREDICTION SEQUENCE"
        gl_render.render_text(
            f"Frame number: {frame} | {status}",
            pos=[0.05*w, 0.95*h],
            font=GLUT_BITMAP_TIMES_ROMAN_24,
        )


def main(args):
    state = {
        "motions": [],
        "v_up_env": utils.str_to_axis(args.axis_up),
        "time_checker_auto_play": viz_utils.TimeChecker(),
        "cur_time": 0.0,
        "play_speed": args.speed,
        "file_idx": 0,
        "time_checker_global": viz_utils.TimeChecker(),
        "args": args,
    }

    v_up_env = state["v_up_env"]

    motions = [
        bvh.load(
            file=filename,
            v_up_skel=utils.str_to_axis(args.axis_up),
            v_face_skel=utils.str_to_axis(args.axis_face),
            v_up_env=v_up_env,
            scale=args.scale,
        )
        for filename in args.bvh_files
    ]

    for i in range(len(motions)):
        operations.translate(motions[i], [args.x_offset * i, 0, 0])

    state["motions"] = motions

    cam = camera.Camera(
        pos=2 * utils.str_to_axis(args.axis_face) + 1 * utils.str_to_axis(args.axis_up),
        origin=np.array([0, 0, 0]),
        vup=v_up_env,
        fov=45.0,
    )
    viewer.run(
        title="Motion Graph Viewer",
        cam=cam,
        size=(1280, 720),
        keyboard_callback=partial(keyboard_callback, state),
        render_callback=partial(render_callback, state),
        idle_callback=partial(idle_callback, state),
        overlay_callback=partial(overlay_callback, state),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BVH file with block body")
    parser.add_argument("--bvh-files", type=str, nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--axis-up", type=str, choices=["x", "y", "z"], default="z")
    parser.add_argument("--axis-face", type=str, choices=["x", "y", "z"], default="y")
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
    main(args)
