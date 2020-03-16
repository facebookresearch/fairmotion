import numpy as np


def upsample_motion_sequence(
    T, current_hertz, desired_hertz, joint_rotations, joint_positions
):
    upsample_rate = int(desired_hertz / current_hertz)
    T_upsampled = T * upsample_rate
    J_p, J_r = joint_positions.shape[1], joint_rotations.shape[1]
    num_position_coordinates, num_rotation_coordinates = (
        joint_positions.shape[2],
        joint_rotations.shape[2],
    )

    joint_rotations_upsampled = np.zeros(
        shape=(T_upsampled, J_r, num_rotation_coordinates)
    )
    joint_positions_upsampled = np.zeros(
        shape=(T_upsampled, J_p, num_position_coordinates)
    )

    for t in range(T):

        t_upsampled = t * upsample_rate
        joint_rotations_upsampled[t_upsampled] = joint_rotations[t]
        joint_positions_upsampled[t_upsampled] = joint_positions[t]

        if (t + 1) < T:
            interpolated_rot_increment = (
                joint_rotations[t + 1] - joint_rotations[t]
            ) / upsample_rate
            interpolated_pos_increment = (
                joint_positions[t + 1] - joint_positions[t]
            ) / upsample_rate
            for i in range(1, upsample_rate):
                joint_rotations_upsampled[t_upsampled + i] = (
                    joint_rotations[t] + i * interpolated_rot_increment
                )
                joint_positions_upsampled[t_upsampled + i] = (
                    joint_positions[t] + i * interpolated_pos_increment
                )
            pass

    return joint_rotations_upsampled, joint_positions_upsampled


def decompose_axis_angles(axis_angles):
    """
    Input dimensions: T x J x num-coordinates
    Output dimensions:
        angles: T x J
        axes: T x J x num-coords
    """

    angles = np.linalg.norm(get_tensor_data(axis_angles), axis=2)
    axes = np.zeros_like(axis_angles)

    for t in range(axis_angles.shape[0]):  # traverse through time/datapoints
        for j in range(axis_angles.shape[1]):  # traverse through joints
            if angles[t, j] < 1e-5:
                axes[t, j, :] = [1, 0, 0]
            else:
                axes[t, j, :] = axis_angles[t, j, :] / angles[t, j]
    return angles, axes
