'''
This script is used to control spot from a provided h5 dataset file.
It uses the absolute positions to control the arm

Author: Moniruzzaman Akash
'''

import os
import h5py, time, math
import numpy as np
from spot_teleop.spot_controller import SpotRobotController
from bosdyn.client.math_helpers  import SE3Pose, Quat

robot_ip = os.environ.get("SPOT_ROBOT_IP", "192.168.1.138")
user     = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")

print(f"Connecting to Spot at {robot_ip} ...")
print(f"user: {user}, password: {password}")

controller = SpotRobotController(robot_ip, user, password, default_exec_time=0.30)

controller.undock()


def inverse_pose7D(pose):
    """Inverse a 7D pose (x, y, z, qx, qy, qz, qw).
    
    Given pose P = (R, t) with quaternion q=(qx,qy,qz,qw) and translation t=(x,y,z),
    return P^{-1} = (R^T, -R^T t) encoded as (x', y', z', qx', qy', qz', qw').
    """
    x, y, z, qx, qy, qz, qw = map(float, pose)
    t = np.array([x, y, z], dtype=float)

    q = np.array([qx, qy, qz, qw], dtype=float)
    # Normalize quaternion to be safe
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")
    q /= norm

    def q_conj(q):
        # (x,y,z,w) -> (-x,-y,-z,w)
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

    def q_mul(q1, q2):
        # Hamilton product for (x,y,z,w) ordering
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=float)

    def rotate_vec(q, v):
        # Rotate vector v by unit quaternion q: v' = q * (v,0) * q_conj
        qv = np.array([v[0], v[1], v[2], 0.0], dtype=float)
        return q_mul(q_mul(q, qv), q_conj(q))[:3]

    q_inv = q_conj(q)                  # rotation inverse = conjugate
    Rt_t  = rotate_vec(q_inv, t)       # R^T * t  (rotate by q_inv)
    t_inv = -Rt_t                      # -R^T * t

    return np.array([t_inv[0], t_inv[1], t_inv[2], q_inv[0], q_inv[1], q_inv[2], q_inv[3]])




# Open the HDF5 file
try:
    hdf5_file_path = "sweep_clean.h5"
    with h5py.File(hdf5_file_path, 'r') as f:
        # Read the data
        demos = f['data']
        demo = demos['demo_0']
        positions = demo['obs']['eef_pos']
        orientations = demo['obs']['eef_quat']
        timestamps = demo['obs']['t']
        gripper_states = demo['obs']['gripper']
        body_poses = demo['obs']['vision_in_body']

        body_pose = inverse_pose7D(body_poses[0])
        print("Body pose (x,y,z,qx,qy,qz,qw): ", body_pose) 
        quat = body_poses[0][3:]

        yaw = math.atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                         1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
        print("Body yaw (deg): ", math.degrees(yaw))
        # controller.move_base_to_pose(body_pose, timeout=5)

        # time.sleep(4)

        controller.reset_pose()  # Move to default pose
        time.sleep(4) # wait for the robot to get to the default pose

        # for i in range(len(positions)):
        #     position = positions[i]
        #     orientation = orientations[i]
        #     timestamp = float(timestamps[i][0])
        #     gripper_state = float(gripper_states[i][0])

        #     # Send the command to the robot
        #     controller.move_arm_to(pos_xyz=position,
        #                             quat_xyzw=orientation,
        #                             gripper=gripper_state)  # Example gripper value

        #     # Print the current step
        #     print(f"Step {i+1}/{len(positions)}: Timestamp {timestamp}, Position {position}, Orientation {orientation}")

        #     # wait for timestamp amount of time
        #     if i < len(timestamps) - 1:
        #         next_timestamp = float(timestamps[i + 1][0])
        #         wait_time = next_timestamp - timestamp
        #         time.sleep(wait_time)
except Exception as e:
    print(f"An error occurred while reading the HDF5 file: {e}")

time.sleep(1) # wait for the robot to undock
controller.stow_arm()
controller.dock()