import numpy as np

from mocap_processing.motion.pfnn import Animation
from mocap_processing.motion.pfnn.Quaternions import Quaternions


def positions_wrt_base(anim):

    # global transforms for each frame F and joint J
    globals = Animation.transforms_global(anim)  # (F, J, 4, 4) ndarray
    transforms_wrt_root = Animation.transforms_blank(anim)

    for i in range(1, anim.shape[1]):  # modifies all joints except root/base
        transforms_wrt_root[:, i] = Animation.transforms_multiply(
            Animation.transforms_inv(globals[:, 0]), globals[:, i]
        )
    transforms_wrt_root[:, 0] = globals[:, 0]  # root/base joint retains global position

    positions_wrt_root = transforms_wrt_root[:, :, 0:3, 3]
    rotations_wrt_root = Quaternions.from_transforms(transforms_wrt_root).angle_axis()

    return rotations_wrt_root, positions_wrt_root


def extract_sequences(
    joint_rotations,
    joint_positions,
    frametime,
    stride=0.25,
    downsampled_hertz=30,
    window_length=5.0,
):

    # initialize data structs, using CNN-like computation of size of input
    J_p, J_r = (
        joint_positions.shape[1],
        joint_rotations.shape[1],
    )  # num of joints
    animation_time = len(joint_rotations) * frametime
    n = int(np.trunc((animation_time - window_length) / stride)) + 1  # sample size
    num_joint_coordinates = joint_rotations.shape[2] + joint_positions.shape[2]
    num_posture_dimensions = (J_p * joint_positions.shape[2]) + (
        J_r * joint_rotations.shape[2]
    )  # J * num_joint_coordinates
    T = int(np.trunc(downsampled_hertz * window_length))

    datapoint_size = num_posture_dimensions * T
    datapoints = np.zeros((n, datapoint_size), dtype=float)
    datapoints_temporal = np.zeros((n, T, num_posture_dimensions), dtype=float)

    current_frame_iterator = 0
    input_hertz = int(np.trunc(1.0 / frametime))
    step_size = int(np.trunc(input_hertz / downsampled_hertz))
    sample_joint_positions = np.zeros((n, T, J_p, joint_positions.shape[2]))
    sample_joint_rotations = np.zeros((n, T, J_r, joint_rotations.shape[2]))

    for i in range(n):
        # retrieve downsampled set of frames, for extracting character body posture over duration of sequence
        posture_sequence_concatenated = []
        posture_sequence_by_time = np.zeros((T, num_posture_dimensions))
        for t in range(
            current_frame_iterator, (T * step_size + current_frame_iterator), step_size
        ):

            # get posture information for given (target) frame
            posture = []
            t_downsampled = int((t - current_frame_iterator) / float(step_size))

            if J_p == J_r:
                J = J_p
                for j in range(J):  # all joints
                    # rotation data
                    sample_joint_rotations[i][t_downsampled][j] = joint_rotations[t][j]
                    posture.extend(joint_rotations[t][j])
                for j in range(J):  # all joints
                    # position data
                    sample_joint_positions[i][t_downsampled][j] = joint_positions[t][j]
                    posture.extend(joint_positions[t][j])
            else:
                joints_pos, joints_rot = set(list(range(J_p))), set(list(range(J_r)))
                J = joints_pos | joints_rot  # union of sets
                for j in J:
                    if j in joints_pos:  # joints considered for position data
                        sample_joint_positions[i][t_downsampled][j] = joint_positions[
                            t
                        ][j]
                        posture.extend(joint_positions[t][j])
                    if j in joints_rot:  # joints considered for rotation data
                        sample_joint_rotations[i][t_downsampled][j] = joint_rotations[
                            t
                        ][j]
                        posture.extend(joint_rotations[t][j])

            posture_sequence_concatenated.extend(posture)
            posture_sequence_by_time[t_downsampled] = posture

        # update datapoints struct with current iteration input
        datapoints[i] = posture_sequence_concatenated
        datapoints_temporal[i] = posture_sequence_by_time
        current_frame_iterator += int(np.trunc(stride * input_hertz))

    return (
        datapoints,
        datapoints_temporal,
        sample_joint_rotations,
        sample_joint_positions,
    )
