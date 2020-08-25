import torch
import numpy as np
from fairmotion.core import motion as motion_class
from fairmotion.core.motion import Joint
from fairmotion.ops import conversions
from fairmotion.utils import constants

from mpl_toolkits.mplot3d import Axes3D

def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx

def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {
    'root': Joint('root', direction=np.zeros(3), length=0, axis=np.zeros(3), dof=[], limits=[])
    }

  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction=direction,
      length=length,
      axis=axis,
      dof=dof,
      limits=limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].child_joints.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent_joint = joints[line[0]]

  return joints

def set_rotation(joint):
  if 'root' in joint.name:
    joint.matrix = joint.C.dot(conversions.E2R(joint.degree)).dot(joint.Cinv)
  else:
    joint.matrix = joint.C.dot(conversions.E2R(joint.degree)).dot(joint.Cinv)
    joint.coordinate = joint.length * joint.matrix.dot(joint.direction)
  for child in joint.child_joints:
    set_rotation(child)

def parse_amc(file_path, joints, skel):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  motion = motion_class.Motion(skel=skel)
  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  frame = 0
  translation_data = []  
  while not EOF:
    # joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      line_idx = 1
      if 'root' in line[0]:
        degree = np.array([float(line[i]) for i in range(4, 7)])
        joints[line[0]].coordinate = np.array([float(line[i]) for i in range(1, 4)])
      else:
        degree = []
        for lm in joints[line[0]].limits:
          if lm[0] != lm[1]:
            degree.append(float(line[line_idx]))
            line_idx += 1
          else:
            degree.append(0)
      joints[line[0]].degree = np.deg2rad(np.array(degree).squeeze())
    pose_data = []
    set_rotation(joints['root'])
    for key in joints.keys():
      if joints[key].matrix is None:
        pose_data.append(constants.eye_T())
      else:
        pose_data.append(conversions.Rp2T(joints[key].matrix.squeeze(), joints[key].coordinate.squeeze()))

    fps = 60
    motion.add_one_frame(frame / fps, pose_data)
    frame += 1
  return motion

def load(file, motion=None, scale=1.0, load_skel=True, load_motion=True):
    if load_skel:
        tmp_joints = parse_asf(file)
        
        # convert format
        joints = []
        parent_joints = []
        for k, v in tmp_joints.items():
            joint = motion_class.Joint(name=k)
            if v.parent_joint is None:
                joint.info["dof"] = 6
                parent_joint = None
                joint.xform_from_parent_joint = conversions.p2T(np.zeros(3))
            else:
                joint.info["dof"] = 3
                parent_joint = v.parent_joint
                joint.xform_from_parent_joint = conversions.p2T(
                    v.direction.squeeze() * v.length
                )
            joints.append(joint)
            parent_joints.append(parent_joint)

        skel = motion_class.Skeleton()
        for i in range(len(joints)):
            skel.add_joint(joints[i], parent_joints[i])

        if load_motion:
            return parse_amc(motion, tmp_joints, skel)
        return skel
    raise NotImplementedError

def save(motion, filename, scale=1.0):
    raise NotImplementedError("Using bvh.save() is recommended")