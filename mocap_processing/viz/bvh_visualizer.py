import argparse
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from basecode.math import mmMath
from basecode.render import camera, gl_render, glut_viewer as viewer
from basecode.utils import basics
from mocap_processing.motion import kinematics


def keyboard_callback(key):
    global viewer, motions, cur_time, w
    global time_checker_auto_play
    global args, file_idx
    global play_speed

    motion = motions[file_idx]

    if key == b"r":
        cur_time = 0.0
        time_checker_auto_play.begin()
        time_checker_global.begin()
    elif key == b"]":
        # print("---------")
        # print(cur_time, motion.time_to_frame(cur_time))
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
        # print("play_speed:", play_speed)
    elif key == b"-":
        play_speed = max(play_speed - 0.2, 0.2)
        # print("play_speed:", play_speed)
    elif key == b"h":
        pose = motion.get_pose_by_frame(motion.time_to_frame(cur_time))
        heights = []
        for j in pose.skel.joints:
            p = mmMath.T2p(pose.get_transform(j, local=False))
            p = mmMath.projectionOnVector(p, pose.skel.v_up_env)
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
        try:
            end_time = float(end_time)
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
        cur_time = start_time
        dt = 1 / 30.0
        while cur_time <= end_time:
            name = "screenshot_%04d" % (cnt_screenshot)
            p = mmMath.T2p(motion.get_pose_by_time(cur_time).get_root_transform())
            viewer.drawGL()
            viewer.save_screen(dir=save_dir, name=name)
            print("\rtime_elased:", cur_time, "(", name, ")", end=" ")
            cur_time += dt
            cnt_screenshot += 1
    else:
        return False
    return True


def render_pose(pose, body_model, color):
    skel = pose.skel
    for j in skel.joints:
        T = pose.get_transform(j, local=False)
        pos = mmMath.T2p(T)
        gl_render.render_point(pos, radius=0.03, color=color)
        if j.parent_joint is not None:
            # returns X that X dot vec1 = vec2
            pos_parent = mmMath.T2p(pose.get_transform(j.parent_joint, local=False))
            p = 0.5 * (pos_parent + pos)
            l = np.linalg.norm(pos_parent - pos)
            r = 0.05
            R = mmMath.getSO3FromVectors(np.array([0, 0, 1]), pos_parent - pos)
            gl_render.render_capsule(mmMath.Rp2T(R, p), l, r, color=color, slice=16)


def render_characters(motions, cur_time, colors):
    global v_up_env
    # glRotatef(90, 1, 0, 0)
    for i, motion in enumerate(motions):
        time_offset = 0.0
        t = (cur_time + time_offset) % motion.length()
        skel = motion.skel
        pose = motion.get_pose_by_frame(motion.time_to_frame(t))
        color = colors[i % len(colors)]

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_LIGHTING)
        render_pose(pose, "stick_figure2", color)


def render_callback():
    global motions, cur_time
    global tex_id_ground

    gl_render.render_ground(
        size=[100, 100], color=[0.8, 0.8, 0.8], axis="z", origin=True, use_arrow=True
    )
    colors = [
        np.array([123, 174, 85, 255]) / 255.0,  # green
        np.array([85, 160, 173, 255]) / 255.0,  # blue
    ]
    render_characters(motions, cur_time, colors)


def idle_callback():
    global viewer, cur_time
    global time_checker_auto_play
    global motions
    global play_speed

    time_elapsed = time_checker_auto_play.get_time(restart=False)
    cur_time += play_speed * time_elapsed
    time_checker_auto_play.begin()


def main(args):
    global motions, cur_time, v_up_env, time_checker_auto_play, play_speed, file_idx

    v_up_env = kinematics.str_to_axis("z")
    motions = [
        kinematics.Motion(
            file=filename,
            v_up_skel=kinematics.str_to_axis(args.axis_up),
            v_face_skel=kinematics.str_to_axis(args.axis_face),
            v_up_env=v_up_env,
            scale=args.scale,
        )
        for filename in args.bvh_files
    ]

    for i in range(len(motions)):
        motions[i].translate([args.x_offset * i, 0, 0])

    cur_time = 0.0
    play_speed = 1.0
    file_idx = 0
    time_checker_auto_play = basics.TimeChecker()
    time_checker_global = basics.TimeChecker()

    cam = camera.Camera(
        pos=np.array([0, 4, 0.5]),
        origin=np.array([0, 0, 0]),
        vup=np.array([0.0, 0.0, 1.0]),
        fov=45.0,
    )
    viewer.run(
        title="Motion Graph Viewer",
        cam=cam,
        size=(1280, 720),
        keyboard_callback=keyboard_callback,
        render_callback=render_callback,
        idle_callback=idle_callback,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize BVH file with block body")
    parser.add_argument("--bvh-files", type=str, nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--axis-up", type=str, choices=["x", "y", "z"], default="z")
    parser.add_argument("--axis-face", type=str, choices=["x", "y", "z"], default="y")
    parser.add_argument(
        "--x-offset",
        type=int,
        default=2,
        help="Translates each"
        " character by x-offset*idx to display them simultaneously side-by-side",
    )
    args = parser.parse_args()
    main(args)
