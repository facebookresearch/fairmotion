# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np

from fairmotion.core import motion as motion_classes
from fairmotion.utils import constants, utils
from fairmotion.ops import conversions


def load(
    file,
    motion=None,
    scale=1.0,
    load_skel=True,
    load_motion=True,
    v_up_skel=np.array([0.0, 1.0, 0.0]),
    v_face_skel=np.array([0.0, 0.0, 1.0]),
    v_up_env=np.array([0.0, 1.0, 0.0]),
):
    if not motion:
        motion = motion_classes.Motion()
    words = None
    with open(file, "rb") as f:
        words = [word.decode() for line in f for word in line.split()]
        f.close()
    assert words is not None and len(words) > 0
    cnt = 0
    total_depth = 0
    joint_stack = [None, None]
    joint_list = []
    parent_joint_list = []

    if load_skel:
        assert motion.skel is None
        motion.skel = motion_classes.Skeleton(
            v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env,
        )

    if load_skel:
        while cnt < len(words):
            # joint_prev = joint_stack[-2]
            joint_cur = joint_stack[-1]
            word = words[cnt].lower()
            if word == "root" or word == "joint":
                parent_joint_list.append(joint_cur)
                name = words[cnt + 1]
                joint = motion_classes.Joint(name=name)
                joint_stack.append(joint)
                joint_list.append(joint)
                cnt += 2
            elif word == "offset":
                x, y, z = (
                    float(words[cnt + 1]),
                    float(words[cnt + 2]),
                    float(words[cnt + 3]),
                )
                T1 = conversions.p2T(scale * np.array([x, y, z]))
                joint_cur.xform_from_parent_joint = T1
                cnt += 4
            elif word == "channels":
                ndofs = int(words[cnt + 1])
                if ndofs == 6:
                    joint_cur.info["type"] = "free"
                elif ndofs == 3:
                    joint_cur.info["type"] = "ball"
                elif ndofs == 1:
                    joint_cur.info["type"] = "revolute"
                else:
                    raise Exception("Undefined")
                joint_cur.info["dof"] = ndofs
                joint_cur.info["bvh_channels"] = []
                for i in range(ndofs):
                    joint_cur.info["bvh_channels"].append(words[cnt + 2 + i].lower())
                cnt += ndofs + 2
            elif word == "end":
                joint_dummy = motion_classes.Joint(name="END")
                joint_stack.append(joint_dummy)
                cnt += 2
            elif word == "{":
                total_depth += 1
                cnt += 1
            elif word == "}":
                joint_stack.pop()
                total_depth -= 1
                cnt += 1
                if total_depth == 0:
                    for i in range(len(joint_list)):
                        motion.skel.add_joint(
                            joint_list[i], parent_joint_list[i],
                        )
                    break
            elif word == "hierarchy":
                cnt += 1
            else:
                raise Exception(f"Unknown Token {word} at token {cnt}")

    if load_motion:
        assert motion.skel is not None
        assert np.allclose(motion.skel.v_up, v_up_skel)
        assert np.allclose(motion.skel.v_face, v_face_skel)
        assert np.allclose(motion.skel.v_up_env, v_up_env)
        while cnt < len(words):
            word = words[cnt].lower()
            if word == "motion":
                num_frames = int(words[cnt + 2])
                dt = float(words[cnt + 5])
                motion.fps = round(1 / dt)
                cnt += 6
                t = 0.0
                range_num_dofs = range(motion.skel.num_dofs)
                for i in range(num_frames):
                    raw_values = [float(words[cnt + j]) for j in range_num_dofs]
                    cnt += motion.skel.num_dofs
                    cnt_channel = 0
                    pose_data = []
                    for joint in motion.skel.joints:
                        T = constants.eye_T()
                        for channel in joint.info["bvh_channels"]:
                            value = raw_values[cnt_channel]
                            if channel == "xposition":
                                value = scale * value
                                T = np.dot(T, conversions.p2T([value, 0, 0]))
                            elif channel == "yposition":
                                value = scale * value
                                T = np.dot(T, conversions.p2T([0, value, 0]))
                            elif channel == "zposition":
                                value = scale * value
                                T = np.dot(T, conversions.p2T([0, 0, value]))
                            elif channel == "xrotation":
                                value = value * np.pi / 180.0
                                T = np.dot(T, conversions.R2T(conversions.Ax2R(value)))
                            elif channel == "yrotation":
                                value = value * np.pi / 180.0
                                T = np.dot(T, conversions.R2T(conversions.Ay2R(value)))
                            elif channel == "zrotation":
                                value = value * np.pi / 180.0
                                T = np.dot(T, conversions.R2T(conversions.Az2R(value)))
                            else:
                                raise Exception("Unknown Channel")
                            cnt_channel += 1
                        pose_data.append(T)
                    motion.add_one_frame(t, pose_data)
                    t += dt
            else:
                cnt += 1
        assert motion.num_frames() > 0
    return motion


def _write_hierarchy(motion, file, joint, scale=1.0, rot_order="XYZ", tab=""):
    def rot_order_to_str(order):
        if order == "xyz" or order == "XYZ":
            return "Xrotation Yrotation Zrotation"
        elif order == "zyx" or order == "ZYX":
            return "Zrotation Yrotation Xrotation"
        else:
            raise NotImplementedError
    joint_order = [joint.name]
    is_root_joint = joint.parent_joint is None
    if is_root_joint:
        file.write(tab + "ROOT %s\n" % joint.name)
    else:
        file.write(tab + "JOINT %s\n" % joint.name)
    file.write(tab + "{\n")
    R, p = conversions.T2Rp(joint.xform_from_parent_joint)
    p *= scale
    file.write(tab + "\tOFFSET %f %f %f\n" % (p[0], p[1], p[2]))
    if is_root_joint:
        file.write(
            tab + "\tCHANNELS 6 Xposition Yposition Zposition %s\n" % \
            rot_order_to_str(rot_order)
        )
    else:
        file.write(tab + "\tCHANNELS 3 %s\n"%rot_order_to_str(rot_order))
    for child_joint in joint.child_joints:
        child_joint_order = _write_hierarchy(
            motion, file, child_joint, scale, rot_order, tab + "\t")
        joint_order.extend(child_joint_order)
    if len(joint.child_joints) == 0:
        file.write(tab + "\tEnd Site\n")
        file.write(tab + "\t{\n")
        file.write(tab + "\t\tOFFSET %f %f %f\n" % (0.0, 0.0, 0.0))
        file.write(tab + "\t}\n")
    file.write(tab + "}\n")
    return joint_order


def save(motion, filename, scale=1.0, rot_order="XYZ", verbose=False):
    if verbose:
        print(" >  >  Save BVH file: %s" % filename)
    with open(filename, "w") as f:
        """ Write hierarchy """
        if verbose:
            print(" >  >  >  >  Write BVH hierarchy")
        f.write("HIERARCHY\n")
        joint_order = _write_hierarchy(
            motion, f, motion.skel.root_joint, scale, rot_order)
        """ Write data """
        if verbose:
            print(" >  >  >  >  Write BVH data")
        t_start = 0
        dt = 1.0 / motion.fps
        num_frames = motion.num_frames()
        f.write("MOTION\n")
        f.write("Frames: %d\n" % num_frames)
        f.write("Frame Time: %f\n" % dt)
        t = t_start

        for i in range(num_frames):
            if verbose and i % motion.fps == 0:
                print(
                    "\r >  >  >  >  %d/%d processed (%d FPS)"
                    % (i + 1, num_frames, motion.fps),
                    end=" ",
                )
            pose = motion.get_pose_by_frame(i)

            for joint_name in joint_order:
                joint = motion.skel.get_joint(joint_name)
                R, p = conversions.T2Rp(pose.get_transform(joint, local=True))
                p *= scale
                R1, R2, R3 = conversions.R2E(R, order=rot_order, degrees=True)
                if joint == motion.skel.root_joint:
                    f.write("%f %f %f %f %f %f " % (p[0], p[1], p[2], R1, R2, R3))
                else:
                    f.write("%f %f %f " % (R1, R2, R3))
            f.write("\n")
            t += dt
            if verbose and i == num_frames - 1:
                print(
                    "\r >  >  >  >  %d/%d processed (%d FPS)"
                    % (i + 1, num_frames, motion.fps)
                )
        f.close()


def load_parallel(files, cpus=20, **kwargs):
    return utils.run_parallel(load, files, num_cpus=cpus, **kwargs)
