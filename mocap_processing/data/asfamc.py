import torch
import numpy as np
from mocap_processing.motion import motion as motion_class
from mocap_processing.utils import conversions, constants

from transforms3d.euler import euler2mat
from mpl_toolkits.mplot3d import Axes3D

class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.
    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.
    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.
    length: Length of the bone.
    axis: Axis of rotation for the bone.
    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.
    limits: Limits on each of the channels in the dof specification
    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')
    plt.show()

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


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
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
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
      direction,
      length,
      axis,
      dof,
      limits
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
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


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
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
    #   joint_degree[line[0]] = [float(deg) for deg in line[1:]]
      degree = []
      line_idx = 1
      for lm in joints[line[0]].limits:
        if lm[0] != lm[1]:
          degree.append(float(line[line_idx]))
          line_idx += 1
        else:
          degree.append(0)
      T = conversions.R2T(
          conversions.A2R(
          np.array(degree, dtype=float)
      ))
      joint_degree[line[0]] = T
    pose_data = []
    for key in joints.keys():
        if key in joint_degree:
            pose_data.append(joint_degree[key])
        else:
            pose_data.append(constants.eye_T())
    # frames.append(pose_data)
    # TODO:  fps unknown
    fps = 1
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
            if v.parent is None:
                joint.info["dof"] = 6
                parent_joint = None
                joint.xform_from_parent_joint = conversions.p2T(np.zeros(3))
            else:
                joint.info["dof"] = 3
                parent_joint = v.parent
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
