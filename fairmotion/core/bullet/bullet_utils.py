import numpy as np
from fairmotion.ops import quaternion

'''
Bullet uses 'xyzw' order for the quaternion. quat_out_order lets
the bullet_utils interprets the order for the application correctly.
'''
xyzw_in = True

def set_base_pQvw(pb_client, body_id, p, Q, v=None, w=None):
    ''' 
    Set positions, orientations, linear and angular velocities of the base link.
    ''' 
    if not xyzw_in:
        Q = quaternion.Q_op(Q, op=['change_order'], xyzw_in=False)
    pb_client.resetBasePositionAndOrientation(body_id, p, Q)
    if v is not None and w is not None:
        pb_client.resetBaseVelocity(body_id, v, w)

def get_base_pQvw(pb_client, body_id):
    ''' 
    Returns position, orientation, linear and angular velocities of the base link.
    ''' 
    p, Q = pb_client.getBasePositionAndOrientation(body_id)
    p, Q = np.array(p), np.array(Q)
    if not xyzw_in:
        Q = quaternion.Q_op(Q, op=['change_order'], xyzw_in=True)
    
    v, w = pb_client.getBaseVelocity(body_id)
    v, w = np.array(v), np.array(w)
    return p, Q, v, w

def get_link_pQvw(pb_client, body_id, indices=None):
    ''' 
    Returns positions, orientations, linear and angular velocities given link indices.
    Please use get_base_pQvw for the base link.
    ''' 
    if indices is None:
        indices = range(pb_client.getNumJoints(body_id))
    
    num_indices = len(indices)
    assert num_indices > 0
    
    ls = pb_client.getLinkStates(body_id, indices, computeLinkVelocity=True)

    ps = np.array([np.array(ls[j][0]) for j in range(num_indices)])
    if not xyzw_in:
        Qs = np.array([
            quaternion.Q_op(np.array(ls[j][1]), op=['change_order'], xyzw_in=True) \
            for j in range(num_indices)])
    else:
        Qs = np.array([np.array(ls[j][1]) for j in range(num_indices)])
    vs = np.array([np.array(ls[j][6]) for j in range(num_indices)])
    ws = np.array([np.array(ls[j][7]) for j in range(num_indices)])

    # if num_indices == 1:
    #     return ps[0], Qs[0], vs[0], ws[0]
    # else:
    #     return ps, Qs, vs, ws
    return ps, Qs, vs, ws

def get_joint_torques(pb_client, body_id, indices=None):
    ''' 
    This will return joint torques applied during the previous simulation
    '''        
    if indices is None:
        indices = range(pb_client.getNumJoints(body_id))

    num_indices = len(indices)
    assert num_indices > 0

    js = pb_client.getJointStatesMultiDof(body_id, indices)

    tqs = np.array([np.array(js[j][3]) for j in range(num_indices)])

    if num_indices == 1:
        return tqs[0]
    else:
        return tqs

def set_joint_pv(pb_client, body_id, indices, ps, vs):
    ''' 
    Set positions and velocities given joint indices.
    Please note that the values are locally repsented w.r.t. its parent joint
    '''        
    ps_processed = ps.copy()
    for i in range(len(ps_processed)):
        if len(ps_processed[i]) == 4 and not xyzw_in:
            ps_processed[i] = \
                quaternion.Q_op(ps_processed[i], op=['change_order'], xyzw_in=False)
    pb_client.resetJointStatesMultiDof(body_id, indices, ps_processed, vs)

def get_joint_pv(pb_client, body_id, indices=None):
    ''' 
    Return positions and velocities given joint indices.
    Please note that the values are locally repsented w.r.t. its parent joint
    '''        
    if indices is None:
        indices = range(pb_client.getNumJoints(body_id))

    num_indices = len(indices)
    assert num_indices > 0

    js = pb_client.getJointStatesMultiDof(body_id, indices)

    ps = []
    vs = []
    for j in range(num_indices):
        p = np.array(js[j][0])
        v = np.array(js[j][1])
        if len(p) == 4 and not xyzw_in:
            p = quaternion.Q_op(p, op=['change_order'], xyzw_in=True)
        ps.append(p)
        vs.append(v)

    # if num_indices == 1:
    #     return ps[0], vs[0]
    # else:
    return ps, vs

def get_state_all(pb_client, body_id):
    ''' 
    Return all state information of the given body. This includes 
    pQvw of the base link and p and v for the all joints
    '''        
    p, Q, v, w = get_base_pQvw(pb_client, body_id)
    ps, vs = get_joint_pv(pb_client, body_id)
    return [p, Q, v, w, ps, vs]

def set_state_all(pb_client, body_id, states):
    ''' 
    Set all state information of the given body. States should include
    pQvw of the base link and p and v for the all joints
    '''        
    p, Q, v, w, ps, vs = states
    assert pb_client.getNumJoints(body_id) == len(ps)

    set_base_pQvw(pb_client, body_id, p, Q, v, w)

    indices = range(len(ps))
    ''' Handling fixed joints '''
    for i in indices:
        if len(ps[i])==0: ps[i] = [0]
        if len(vs[i])==0: vs[i] = [0]
    set_joint_pv(pb_client, body_id, indices, ps, vs)

def get_mass(pb_client, body_id, indices=None):
    '''
    Return masses of the links
    '''
    if indices is None:
        indices = range(-1, pb_client.getNumJoints(body_id))
    masses = []
    for i in indices:
        di = pb_client.getDynamicsInfo(body_id, i)
        masses.append(di[0])
    return masses

def compute_com_and_com_vel(pb_client, body_id, indices=None):
    '''
    Return the center-of-mass and the center-of-mass velocity
    '''
    if indices is None:
        indices = range(-1, pb_client.getNumJoints(body_id))

    total_mass = 0.0
    com = np.zeros(3)
    com_vel = np.zeros(3)
    
    for i in indices:
        di = pb_client.getDynamicsInfo(body_id, i)
        mass = di[0]
        if i==-1:
            p, _, v, _ = get_base_pQvw(pb_client, body_id)
        else:
            ls = pb_client.getLinkState(body_id, i, computeLinkVelocity=True)
            p, v = np.array(ls[0]), np.array(ls[6])
        total_mass += mass
        com += mass * p
        com_vel += mass * v
    com /= total_mass
    com_vel /= total_mass
    return com, com_vel

def _compute_com_and_com_vel(pb_client, body_id, indices=None, masses=None):
    if indices is None:
        indices = range(-1, pb_client.getNumJoints(body_id))
    if masses is None:
        masses = get_mass(pb_client, body_id, indices)
    assert len(indices)==len(masses)

    total_mass = 0.0
    com = np.zeros(3)
    com_vel = np.zeros(3)

    indices_wo_base = []
    masses_wo_base = []
    for i in range(len(indices)):
        if indices[i]<0:
            p, _, v, _ = get_base_pQvw(pb_client, body_id)
            mass = masses[i]
            total_mass += mass
            com += mass * p
            com_vel = mass * v
        else:
            indices_wo_base.append(indices[i])
            masses_wo_base.append(masses[i])

    ls = pb_client.getLinkStates(body_id, indices_wo_base, computeLinkVelocity=True)

    for i in range(len(masses_wo_base)):
        mass = masses_wo_base[i]
        p, v = np.array(ls[i][0]), np.array(ls[i][6])
        total_mass += mass
        com += mass * p
        com_vel = mass * v

    com /= total_mass
    com_vel /= total_mass
    return com, com_vel

def compute_PD_forces(pb_client, 
                      body_id, 
                      joint_indices, 
                      desired_positions, 
                      desired_velocities, 
                      kps, 
                      kds,
                      max_forces):
    '''
    Compute PD forces given target values (P and D) and the simulated agent.
    This was implented because PD_CONTROL for setJointMotorControlMultiDofArray
    in PyBullet does not support spherical joint yet.
    '''
    forces = []
    js = pb_client.getJointStatesMultiDof(body_id, joint_indices)
    for i in range(len(joint_indices)):
        joint_pos = js[i][0]
        joint_vel = js[i][1]
        # print(i, desired_positions[i], joint_pos)
        if len(joint_pos) == 1:
            desired_pos = desired_positions[i]
            desired_vel = desired_velocities[i]
            qerror = desired_pos - joint_pos[0]
            qdoterror = desired_vel - joint_vel[0]
        elif len(joint_pos) == 4:
            desired_pos = desired_positions[i]
            desired_vel = desired_velocities[i]
            qerror = np.array(pb_client.getAxisDifferenceQuaternion(desired_pos, joint_pos))
            qdoterror = np.array(desired_vel - joint_vel)
        else:
            raise NotImplementedError
        f = kps[i] * qerror + kds[i] * qdoterror
        f = np.clip(f, -max_forces[i], max_forces[i])
        forces.append(f)
    return forces
