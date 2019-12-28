import copy
import gzip
import math
import numpy as np
import pickle

from basecode.utils import basics
from basecode.utils import multiprocessing as mp
from basecode.math import mmMath

EPS = 1.0e-6

def get_index(index_dict, key):
    if isinstance(key, int):
        return key
    elif isinstance(key, str):
        return index_dict[key]
    else:
        return index_dict[key.name]


class Joint(object):
    def __init__(self, name="joint"):
        self.name = name
        self.parent_joint = None
        self.child_joint = []
        self.index_child_joint = {}
        self.xform_global = mmMath.I_SE3()
        self.xform_from_parent_joint = mmMath.I_SE3()
        self.info = {}

    def set_parent_joint(self, joint):
        assert isinstance(joint, Joint)
        self.parent_joint = joint
        self.xform_global = np.dot(self.parent_joint.xform_global, 
                                   self.xform_from_parent_joint)

    def add_child_joint(self, joint):
        assert isinstance(joint, Joint)
        assert joint.name not in self.index_child_joint.keys()
        self.index_child_joint[joint.name] = len(self.child_joint)
        self.child_joint.append(joint)
        joint.set_parent_joint(self)
    
    def get_child_joint(self, key):
        return self.child_joint[get_index(self.index_child_joint, key)]

    def get_child_joint_recursive(self):
        ''' This could have duplicated joints if there exists loops in the chain'''
        joints = []
        for j in self.child_joint:
            joints.append(j)
            joints += j.get_all_child_joint()
        return joints


class Skeleton(object):
    def __init__(self, 
                 name="skeleton", 
                 v_up=np.array([0.0, 1.0, 0.0]), 
                 v_face=np.array([0.0, 0.0, 1.0]), 
                 v_up_env=np.array([0.0, 1.0, 0.0]), 
                 ):
        self.name = name
        self.joints = []
        self.index_dof_start_end = {}
        self.index_joint = {}
        self.root_joint = None
        self.num_dofs = 0
        self.v_up = v_up
        self.v_face = v_face
        self.v_up_env = v_up_env

    def num_joint(self):
        return len(self.joints)

    def get_index_dof_start_end(self, key):
        return get_index(self.index_dof_start_end, key)

    def get_index_joint(self, key):
        return get_index(self.index_joint, key)

    def get_joint(self, key):
        return self.joints[self.get_index_joint(key)]

    def add_joint(self, joint, parent_joint):
        if parent_joint is None:
            assert self.num_joint()==0
            self.root_joint = joint
        else:
            parent_joint = self.get_joint(parent_joint)
            parent_joint.add_child_joint(joint)
        self.index_joint[joint.name] = len(self.joints)
        self.joints.append(joint)
        self.index_dof_start_end[joint.name] = (self.num_dofs, self.num_dofs+joint.info['dof'])
        self.num_dofs += joint.info['dof']

class Posture(object):
    def __init__(self, skel, data):
        assert isinstance(skel, Skeleton)
        assert skel.num_joint()==len(data)
        self.skel = skel
        self.data = data
  
    def get_transform(self, key, local):
        skel = self.skel
        if local:
            return self.data[skel.get_index_joint(key)]
        else:
            joint = skel.get_joint(key)
            T = np.dot(joint.xform_from_parent_joint, 
                       self.data[skel.get_index_joint(joint)])
            while joint.parent_joint is not None:
                T_j = np.dot(joint.parent_joint.xform_from_parent_joint,
                             self.data[skel.get_index_joint(joint.parent_joint)])
                T = np.dot(T_j, T)
                joint = joint.parent_joint
            return T
    def set_transform(self, key, T, local, do_ortho_norm=True):
        if local:
            T1 = T
        else:
            T0 = self.skel.get_joint(key).xform_global
            T1 = np.dot(mmMath.invertSE3(T0), T)
        if do_ortho_norm:
            ''' 
            This insures that the rotation part of 
            the given transformation is valid 
            '''
            Q,p = mmMath.T2Qp(T1)
            Q = mmMath.post_process_Q(Q, normalize=True, half_space=False)
            T1 = mmMath.Qp2T(Q,p)
        self.data[self.skel.get_index_joint(key)] = T1
  
    def get_root_transform(self):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        return self.get_transform(root_idx, local=False)
  
    def set_root_transform(self, T, local):
        root_idx = self.skel.get_index_joint(self.skel.root_joint)
        self.set_transform(root_idx, T, local)
  
    def translate(self, v, local=False):
        self.transform(mmMath.p2T(v), local)
 
    def rotate(self, R, local=False):
        self.transform(mmMath.R2T(R), local)
  
    def transform(self, T, local=False):
        R0,p0 = mmMath.T2Rp(self.get_root_transform())
        R1,p1 = mmMath.T2Rp(T)
        if local:
            R,p = np.dot(R0, R1), p0+np.dot(R0,p1)
        else:
            R,p = np.dot(R1, R0), p0+p1
        self.set_root_transform(mmMath.Rp2T(R,p), local=False)

    def get_facing_transform(self):
        d, p = self.get_facing_direction_position()
        z = d
        y = self.skel.v_up_env
        x = np.cross(y, z)
        return mmMath.Rp2T(np.array([x, y, z]).transpose(), p)
        
    def get_facing_position(self):
        d, p = self.get_facing_direction_position()
        return p
  
    def get_facing_direction(self):
        d, p = self.get_facing_direction_position()
        return d
  
    def get_facing_direction_position(self):
        R, p = mmMath.T2Rp(self.get_root_transform())
        d = np.dot(R, self.skel.v_face)
        d = d - mmMath.projectionOnVector(d, self.skel.v_up_env)
        p = p - mmMath.projectionOnVector(p, self.skel.v_up_env)
        return d/np.linalg.norm(d), p


class Velocity(object):
    ''' 
    This contains linear and angluar velocity of joints.
    All velocities are represented w.r.t. the joint frame.
    To get the global velocity, you should give the frame 
    that corresponds to the velocity.
    '''
    def __init__(self, pose1=None, pose2=None, dt=None, skel=None):
        self.data_local = []
        self.data_global = []
        if pose1:
            assert pose2 and dt
            assert isinstance(pose1, Posture) and isinstance(pose2, Posture)
            self.skel = pose1.skel
            self._compute(pose1, pose2, dt)
        if skel:
            self.skel = skel
 
    def set(self, skel=None, data_local=None, data_global=None):
        if skel: self.skel = skel
        if data_local: self.data_local = data_local
        if data_global: self.data_global = data_global

    def _compute(self, pose1, pose2, dt):
        assert pose1.skel.num_joint()==pose2.skel.num_joint()
        assert dt > EPS
        for joint in self.skel.joints:
            T1 = pose1.get_transform(joint, local=True)
            T2 = pose2.get_transform(joint, local=True)
            dR,dp = mmMath.T2Rp(np.dot(mmMath.invertSE3(T1), T2))
            w,v = mmMath.logSO3(dR)/dt, dp/dt
            self.data_local.append(np.hstack((w,v)))
            T1 = pose1.get_transform(joint, local=False)
            T2 = pose2.get_transform(joint, local=False)
            R, p = mmMath.T2Rp(np.dot(mmMath.invertSE3(T1), T2))
            w, v = mmMath.logSO3(R)/dt, p/dt
            self.data_global.append(np.hstack((w,v)))
   
    def get_velocity(self, key, local, R_ref=None):
        return np.hstack([self.get_angular_velocity(key, local, R_ref), 
                          self.get_linear_velocity(key, local, R_ref)])
  
    def get_angular_velocity(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        w = data[self.skel.get_index_joint(key)][:3]
        if R_ref is not None: w = np.dot(R_ref, w)
        return w
    
    def get_linear_velocity(self, key, local, R_ref=None):
        data = self.data_local if local else self.data_global
        v = data[self.skel.get_index_joint(key)][3:6]
        if R_ref is not None: v = np.dot(R_ref, v)
        return v
    
    def rotate(self, R):
        data_global_new = []
        for d in self.data_global:
            w = d[:3]
            v = d[3:6]
            w = np.dot(R, w)
            v = np.dot(R, v)
            data_global_new.append(np.hstack([w, v]))
        self.data_global = data_global_new


def interpolate_posture(alpha, posture1, posture2):
    skel = posture1.skel
    data = []
    for j in skel.joints:
        R1,p1 = mmMath.T2Rp(posture1.get_transform(j, local=True))
        R2,p2 = mmMath.T2Rp(posture2.get_transform(j, local=True))
        R,p = mmMath.slerp(R1, R2, alpha), mmMath.linearInterpol(p1, p2, alpha)
        data.append(mmMath.Rp2T(R,p))
    return Posture(posture1.skel, data)

class Motion(object):
    def __init__(self, 
                 name='motion',
                 skel=None, 
                 file=None, 
                 scale=1.0, 
                 load_skel=True, 
                 load_motion=True, 
                 v_up_skel=np.array([0.0, 1.0, 0.0]), 
                 v_face_skel=np.array([0.0, 0.0, 1.0]), 
                 v_up_env=np.array([0.0, 1.0, 0.0]), 
                 ):
        self.name = name
        self.skel = skel
        self.times = []
        self.postures = []
        self.velocities = []
        self.info = {}
        if file: self.load_bvh(file=file,
                               scale=scale,
                               load_skel=load_skel,
                               load_motion=load_motion,
                               v_up_skel=v_up_skel,
                               v_face_skel=v_face_skel,
                               v_up_env=v_up_env,
                               )
    def clear(self):
        self.times = []
        self.postures = []
        self.velocities = []
        self.info = {}

    def set_skeleton(self, skel):
        self.skel = skel
        for i in range(self.num_frame()):
            self.postures[i].skel = skel
            self.velocities[i].skel = skel

    def detach(self, frame_start, frame_end, init_time=True, copied=True):
        motion = Motion(skel=self.skel)
        motion.name = '%s_copied_%d_%d'%(self.name, frame_start, frame_end)
        motion.times = self.times[frame_start:frame_end+1]
        motion.postures = self.postures[frame_start:frame_end+1]
        motion.velocities = self.velocities[frame_start:frame_end+1]
        if copied:
            motion = copy.deepcopy(motion)
            if init_time:
                t_init = motion.times[0]
                for i in range(motion.num_frame()):
                    motion.times[i] -= t_init
        else:
            assert not init_time
        return motion

    def append(self, motion, blend_length=0.5):
        assert isinstance(motion, Motion)
        assert self.skel.num_joint() == motion.skel.num_joint()
        assert motion.num_frame() > 0
        assert motion.length() > blend_length

        motion = copy.deepcopy(motion)
        
        ''' If the current motion is empty, just copy the given motion '''
        if self.num_frame() == 0:
            for i in range(motion.num_frame()):
                self.times.append(motion.times[i]-motion.times[0])
                self.postures.append(motion.postures[i])
                self.velocities.append(motion.velocities[i])
            return
        frame_target = motion.time_to_frame(motion.times[0]+blend_length)
        frame_source = self.time_to_frame(self.times[-1]-blend_length)
        pose1 = self.get_pose_by_frame(frame_source)
        pose2 = motion.postures[0]

        R1,p1 = mmMath.T2Rp(pose1.get_root_transform())
        R2,p2 = mmMath.T2Rp(pose2.get_root_transform())

        ''' Translation to be applied '''
        dp = p1-p2
        dp = dp - mmMath.projectionOnVector(dp, self.skel.v_up_env)
        axis = self.skel.v_up_env

        ''' Rotation to be applied '''
        Q1 = mmMath.R2Q(R1)
        Q2 = mmMath.R2Q(R2)
        Q2, theta = mmMath.nearest_Q(Q1, Q2, axis)
        dR = mmMath.exp(axis, theta)

        motion.translate(dp)
        motion.rotate(dR)
        # motion.transform(mmMath.Rp2T(dR, dp))

        t_total = self.get_time_by_frame(frame_source)
        t_processed = 0.0
        times_new = []
        poses_new = []
        for i in range(1, motion.num_frame()):
            dt = (motion.get_time_by_frame(i)-motion.get_time_by_frame(i-1))
            t_total += dt
            t_processed += dt
            pose_target = motion.get_pose_by_frame(i)
            ''' Blend pose for a moment '''
            if t_processed <= blend_length:
                alpha = t_processed/float(blend_length)
                pose_source = self.get_pose_by_time(t_total)
                # pose_source = self.get_pose_by_frame(frame_source+i)
                # pose_source = self.get_pose_by_frame(frame_source)
                # pose_target = motion.get_pose_by_frame(frame_target)
                pose_target = pose_source.blend(pose_target, alpha)
            times_new.append(t_total)
            poses_new.append(pose_target)
            
        del self.times[frame_source+1:]
        del self.postures[frame_source+1:]
        for i in range(len(poses_new)):
            self.add_one_frame(copy.deepcopy(times_new[i]),
                               copy.deepcopy(poses_new[i].data))

    def add_one_frame(self, t, pose_data, vel_data=None):
        self.times.append(t)
        self.postures.append(Posture(self.skel, pose_data))
        ''' We push a dummy velocity when we have only one frame '''
        if self.num_frame()==1:
            self.velocities.append(Velocity(self.postures[-1], self.postures[-1], 0.01))
        ''' 
        When we add the second frame, 
        we can modify the dummy velocity that we added before 
        '''
        if self.num_frame()==2:
            dt = self.times[-1] - self.times[-2]
            self.velocities[-1] = Velocity(self.postures[-2], self.postures[-1], dt)
        ''' We can compute a real velociy when we have more than two frames '''
        if self.num_frame()>=2:
            dt = self.times[-1] - self.times[-2]
            self.velocities.append(Velocity(self.postures[-2], self.postures[-1], dt))
    
    def frame_to_time(self, frame):
        return self.times[frame]
   
    def time_to_frame(self, time):
        return basics.bisect(self.times, time)
        
    def get_time_by_frame(self, frame):
        assert frame < self.num_frame()
        return self.times[frame]
    
    def get_pose_by_frame(self, frame):
        assert frame < self.num_frame()
        return self.postures[frame]

    def get_pose_by_time(self, time):
        time = basics.clamp(time, self.times[0], self.times[-1])
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1+1, self.num_frame()-1)
        if frame1==frame2: return self.postures[frame1]
        t1 = self.times[frame1]
        t2 = self.times[frame2]
        if time==t1: return self.postures[frame1]
        if time==t2: return self.postures[frame2]
        assert (t2-t1)>1.0e-5
        alpha = basics.clamp((time-t1)/(t2-t1), 0.0, 1.0)
        return interpolate_posture(alpha, self.postures[frame1], self.postures[frame2])

    def get_velocity_by_frame(self, frame):
        assert frame < self.num_frame()
        return self.velocities[frame]

    def get_velocity_by_time(self, time):
        time = basics.clamp(time, self.times[0], self.times[-1])
        frame1 = self.time_to_frame(time)
        frame2 = min(frame1+1, self.num_frame()-1)
        if frame1==frame2: return self.velocities[frame1]
        t1 = self.times[frame1]
        t2 = self.times[frame2]
        if time==t1: return self.velocities[frame1]
        if time==t2: return self.velocities[frame2]
        assert (t2-t1)>EPS
        alpha = basics.clamp((time-t1)/(t2-t1), 0.0, 1.0)
        vel1 = self.get_velocity_by_frame(frame1)
        vel2 = self.get_velocity_by_frame(frame2)
        vel_data_local = []
        vel_data_global = []
        for joint in self.skel.joints:
            v1 = vel1.get_velocity(joint, local=True)
            v2 = vel2.get_velocity(joint, local=True)
            vel_data_local.append(mmMath.linearInterpol(v1, v2, alpha))
            v1 = vel1.get_velocity(joint, local=False)
            v2 = vel2.get_velocity(joint, local=False)
            vel_data_global.append(mmMath.linearInterpol(v1, v2, alpha))
        v = Velocity()
        v.set(self.skel, vel_data_local, vel_data_global)
        return v

    def num_frame(self):
        return len(self.times)

    def length(self):
        return self.times[-1]-self.times[0]

    # translate, rotate, transform gets inputs represented in world space
    def translate(self, v):
        for pose in self.postures:
            pose.translate(v, local=False)

    def rotate(self, R, anchor_frame=0):
        T = mmMath.R2T(R)
        self.transform(T, anchor_frame)

    def transform(self, T, anchor_frame=0):
        ''' Change postures '''
        T_from_anchor_pose = []
        pose_anchor = self.get_pose_by_frame(anchor_frame)
        T_root = pose_anchor.get_root_transform()
        T_root_inv = mmMath.invertSE3(T_root)
        for i in range(self.num_frame()):
            pose = self.get_pose_by_frame(i)
            T_rel = np.dot(T_root_inv, pose.get_root_transform())
            T_from_anchor_pose.append(T_rel)
        pose_anchor.transform(T, local=False)
        T_root_new = pose_anchor.get_root_transform()
        for i in range(self.num_frame()):
            T_rel = T_from_anchor_pose[i]
            T_new = np.dot(T_root_new, T_rel)
            self.get_pose_by_frame(i).set_root_transform(T_new, local=False)
        ''' Change velocity (global value only) '''
        R = mmMath.T2R(T)
        for i in range(self.num_frame()):
            self.get_velocity_by_frame(i).rotate(R)

    def load_bvh(self, file, scale, load_skel, load_motion, v_up_skel, v_face_skel, v_up_env):
        words = None
        with open(file, 'rb') as f:
            words = [word.decode() for line in f for word in line.split()]
            f.close()
        assert words is not None and len(words) > 0
        cnt = 0
        total_depth = 0
        joint_stack = [None, None]
        joint_list = []
        parent_joint_list = []
        
        if load_skel:
            assert self.skel is None
            self.skel = Skeleton(v_up=v_up_skel, v_face=v_face_skel, v_up_env=v_up_env)
        
        if load_skel:
            while cnt < len(words):
                joint_prev = joint_stack[-2]
                joint_cur = joint_stack[-1]
                word = words[cnt].lower()
                if word=='root' or word=='joint':
                    parent_joint_list.append(joint_cur)
                    name = words[cnt+1]
                    joint = Joint(name)
                    joint_stack.append(joint)
                    joint_list.append(joint)
                    cnt += 2
                elif word=='offset':
                    x, y, z = float(words[cnt+1]), float(words[cnt+2]), float(words[cnt+3])
                    T1 = mmMath.getSE3ByTransV(scale * np.array([x, y, z]))
                    joint_cur.xform_from_parent_joint = T1
                    cnt += 4
                elif word=='channels':
                    ndofs = int(words[cnt+1])
                    if ndofs==6:
                        joint_cur.info['type'] = 'free'
                    elif ndofs==3:
                        joint_cur.info['type'] = 'ball'
                    elif ndofs==1:
                        joint_cur.info['type'] = 'revolute'
                    else:
                        raise Exception('Undefined')
                    joint_cur.info['dof'] = ndofs
                    joint_cur.info['bvh_channels'] = []
                    for i in range(ndofs):
                        joint_cur.info['bvh_channels'].append(words[cnt+2+i].lower())
                    cnt += ndofs+2
                elif word=='end':
                    joint_dummy = Joint("END")
                    joint_stack.append(joint_dummy)
                    cnt += 2
                elif word=='{':
                    total_depth += 1
                    cnt += 1
                elif word=='}':
                    joint_stack.pop()
                    total_depth -= 1
                    cnt += 1
                    if total_depth==0:
                        for i in range(len(joint_list)):
                            self.skel.add_joint(joint_list[i], parent_joint_list[i])
                        break
                elif word=='hierarchy':
                    cnt += 1
                else:
                    raise Exception("Unknown Token", word)
        
        if load_motion:
            assert self.skel is not None
            assert np.allclose(self.skel.v_up, v_up_skel)
            assert np.allclose(self.skel.v_face, v_face_skel)
            assert np.allclose(self.skel.v_up_env, v_up_env)
            while cnt < len(words):
                word = words[cnt].lower()
                if word=='motion':
                    num_frame = int(words[cnt+2])
                    dt = float(words[cnt+5])
                    cnt += 6
                    t = 0.0
                    range_num_dofs = range(self.skel.num_dofs)
                    for i in range(num_frame):
                        raw_values = [float(words[cnt+j]) for j in range_num_dofs]
                        cnt += self.skel.num_dofs
                        cnt_channel = 0
                        pose_data = []
                        for joint in self.skel.joints:
                            T = mmMath.I_SE3()
                            for channel in joint.info['bvh_channels']:
                                value = raw_values[cnt_channel]
                                if channel == 'xposition':
                                    value = scale*value
                                    T = np.dot(T, mmMath.getSE3ByTransV([value, 0, 0]))
                                elif channel == 'yposition':
                                    value = scale*value
                                    T = np.dot(T, mmMath.getSE3ByTransV([0, value, 0]))
                                elif channel == 'zposition':
                                    value = scale*value
                                    T = np.dot(T, mmMath.getSE3ByTransV([0, 0, value]))
                                elif channel == 'xrotation':
                                    value = value*math.pi/180.0
                                    T = np.dot(T, mmMath.SO3ToSE3(mmMath.rotX(value)))
                                elif channel == 'yrotation':
                                    value = value*math.pi/180.0
                                    T = np.dot(T, mmMath.SO3ToSE3(mmMath.rotY(value)))
                                elif channel == 'zrotation':
                                    value = value*math.pi/180.0
                                    T = np.dot(T, mmMath.SO3ToSE3(mmMath.rotZ(value)))
                                else:
                                    raise Exception('Unknown Channel')
                                cnt_channel += 1
                            pose_data.append(T)
                        self.add_one_frame(t, pose_data)
                        t += dt
                else:
                    cnt += 1
            assert self.num_frame() > 0
    
    def _write_hierarchy_bvh(self, file, joint, scale=1.0, tab=""):
        is_root_joint = joint.parent_joint is None
        if is_root_joint:
            file.write(tab+"ROOT %s\n"%joint.name)
        else:
            file.write(tab+"JOINT %s\n"%joint.name)
        file.write(tab+"{\n")
        R, p = mmMath.T2Rp(joint.xform_from_parent_joint)
        p *= scale
        file.write(tab+"\tOFFSET %f %f %f\n"%(p[0],p[1],p[2]))
        if is_root_joint:
            file.write(tab+"\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
        else:
            file.write(tab+"\tCHANNELS 3 Zrotation Yrotation Xrotation\n")
        for child_joint in joint.child_joint:
            self._write_hierarchy_bvh(file, child_joint, scale, tab+"\t")
        if len(joint.child_joint)==0:
            file.write(tab+"\tEnd Site\n")
            file.write(tab+"\t{\n")
            file.write(tab+"\t\tOFFSET %f %f %f\n"%(0.0, 0.0, 0.0))
            file.write(tab+"\t}\n")
        file.write(tab+"}\n")

    def save_bvh(self, file_name, fps=30, scale=1.0, verbose=False):
        if verbose:
            print(">> Save BVH file: %s"%file_name)
        with open(file_name, 'w') as f:
            ''' Write hierarchy '''
            if verbose:
                print(">>>> Write BVH hierarchy")
            f.write("HIERARCHY\n")
            self._write_hierarchy_bvh(f, self.skel.root_joint, scale)
            ''' Write data '''
            if verbose:
                print(">>>> Write BVH data")
            t_start = self.times[0]
            t_end = self.times[-1]
            dt = 1.0/fps
            num_frame = int((t_end - t_start) * fps)+1
            f.write("MOTION\n")
            f.write("Frames: %d\n"%num_frame)
            f.write("Frame Time: %f\n"%dt)
            t = t_start
            for i in range(num_frame):
                if verbose and i%fps==0:
                    print("\r>>>> %d/%d processed (%d FPS)"%(i+1, num_frame, fps), end=" ")
                pose = self.get_pose_by_time(t)
                for joint in self.skel.joints:
                    R, p = mmMath.T2Rp(pose.get_transform(joint, local=True))
                    p *= scale
                    v = mmMath.R2ZYX(R)
                    v *= 180.0 / math.pi
                    Rz, Ry, Rx = v[0], v[1], v[2]
                    if joint == self.skel.root_joint:
                        f.write("%f %f %f %f %f %f "%(p[0], p[1], p[2], Rz, Ry, Rx))
                    else:
                        f.write("%f %f %f "%(Rz, Ry, Rx))
                f.write("\n")
                t += dt
                if verbose and i==num_frame-1:
                    print("\r>>>> %d/%d processed (%d FPS)"%(i+1, num_frame, fps))
            f.close()
    
    def resample(self, fps):
        times_new = []
        postures_new = []
        velocities_new = []

        dt = 1.0/fps
        t = self.times[0]
        while t < self.times[-1]:
            pose = self.get_pose_by_time(t)
            vel = self.get_velocity_by_time(t)
            pose.skel = self.skel
            vel.skel = self.skel
            times_new.append(t)
            postures_new.append(pose)
            velocities_new.append(vel)
            t += dt

        self.times = times_new
        self.postures = postures_new
        self.velocities = velocities_new


def read_matrix(string, shape=(3,3)):
    return np.fromstring(string, dtype=float, sep=' ').reshape(shape)


def read_vector(string):
    return np.fromstring(string, dtype=float, sep=' ')


def read_transform(xml_node):
    R = mmMath.I_SO3()
    p = np.zeros(3)
    if 'linear' in xml_node.attrib: 
        R = read_matrix(xml_node.attrib['linear'])
    if 'translation' in xml_node.attrib: 
        p = read_vector(xml_node.attrib['translation'])
    return mmMath.Rp2T(R,p)


def read_bool(string):
    if string=="true" or string=="True": 
        return True
    elif string=="false" or string=="False": 
        return False
    else:
        raise Exception("Unknown Token", string)


def read_sign(string):
    if string=="+": return 1.0
    elif string=="-": return -1.0
    else: raise Exception("Unknown Token", string)


def _read_motions(job_idx, scale, v_up_skel, v_face_skel, v_up_env, resample, fps, verbose):
    res = []
    if job_idx[0] >= job_idx[1]:
        return res
    for i in range(job_idx[0], job_idx[1]):
        file = mp.shared_data[i]
        if file.endswith('.bvh'):
            motion = Motion(file=file,
                            scale=scale,
                            v_up_skel=v_up_skel, 
                            v_face_skel=v_face_skel,
                            v_up_env=v_up_env)
        elif file.endswith('.motion.gzip'):
            with gzip.open(file, "rb") as f:
                motion = pickle.load(f)
        else:
            raise Exception('Unknown Motion File Type')
        if resample: motion.resample(fps=fps)
        if verbose: print('Loaded: %s'%file)
        res.append(motion)
    return res


def read_motion_parallel(files, num_worker, scale, v_up_skel, v_face_skel, v_up_env, resample, fps, verbose):
    ''' 
    Load motion files in parallel
    
    Parameters
    ----------
    files : a list of str
        a list containing motion file names
    num_worker : int
        the number of cpus to use
    scale : float
        scale for loading motion
    v_up_skel : numpy array R^3
        the up vector of skeleton
    v_face_skel : numpy array R^3
        the facing vector of skeleton
    v_up_env : numpy array R^3
        the up vector of the environment
    resample : bool
        whether resampling is performed when loading motions
    fps : int
        FPS (Frames Per Second) used for resampling
    verbose : bool
        if True then print some status messages
    '''
    mp.shared_data = files
    motions = mp.run_parallel_async_idx(_read_motions, 
                                        num_worker, 
                                        len(mp.shared_data),
                                        scale,
                                        v_up_skel,
                                        v_face_skel,
                                        v_up_env,
                                        resample,
                                        fps,
                                        verbose,
                                        )
    return motions