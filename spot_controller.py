#!/usr/bin/env python3
"""
spot_controller.py
-------------------
Utility class to command Boston Dynamics Spot's arm by either

1. apply_action(10-D vector)
     [ Δx Δy Δz,  rot6d(6),  gripper_abs ]

2. send_pose(position_xyz, quaternion_xyzw, gripper_abs)
     absolute command (metres, unit quaternion, 0 -1 gripper)

No Diffusion-Policy code inside: just call apply_action(action) at your control rate.

Requires:
    pip install bosdyn-client

Author: Moniruzzaman Akash
"""
from __future__ import annotations
import time, os, sys, signal, math
from pathlib import Path
import numpy as np

# ───── Spot SDK ─────────────────────────────────────────────────────────── #
from bosdyn.client           import create_standard_sdk
from bosdyn.client.lease     import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, blocking_sit, block_until_arm_arrives, block_for_trajectory_cmd
from bosdyn.client.robot_state    import RobotStateClient
from bosdyn.client.estop   import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers  import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.docking           import DockingClient, blocking_dock_robot, blocking_undock, get_dock_id
from bosdyn.api              import geometry_pb2, arm_command_pb2, trajectory_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.api.spot         import robot_command_pb2 as spot_command_pb2
from google.protobuf import duration_pb2
from bosdyn.client.math_helpers  import SE3Pose, Quat
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.image import ImageClient
from spot_images import SpotImages
from bosdyn.client.exceptions import InternalServerError
from utils.spot_utils import quat_to_matrix, matrix_to_quat, rot6d_to_matrix, image_to_cv
import logging, cv2


# ───── Controller class ─────────────────────────────────────────────────── #
class SpotRobotController:
    def __init__(self,
                 robot_ip: str,
                 username: str,
                 password: str,
                 arm_base_frame: str = BODY_FRAME_NAME,
                 default_exec_time: float = 0.2):
        """
        frame_name : reference frame for arm pose commands.
        default_cmd_time : seconds over which Spot will execute each pose command.
        """
        self.arm_base_frame = arm_base_frame
        self.t_exec     = default_exec_time
        # Integrate deltas on a target pose to decouple from observation latency.
        self._arm_target_pos = None
        self._arm_target_quat = None
        self._arm_target_frame = None

        # --- connect & auth ------------------------------------------------
        sdk   = create_standard_sdk("spot-controller")
        self.robot = sdk.create_robot(robot_ip)
        self.robot.authenticate(username, password)
        self.robot.time_sync.wait_for_sync()

        # --- lease ---------------------------------------------------------
        lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        try:
            lease = lease_client.acquire()
        except ResourceAlreadyClaimedError:
            lease = lease_client.take()
        self.robot.lease_wallet.add(lease)
        self._lease_keepalive = LeaseKeepAlive(lease_client)

        # --- estop ---------------------------------------------------------
        estop_client = self.robot.ensure_client("estop")
        self.estop_endpoint = EstopEndpoint(estop_client, "vr_teleop", 9.0)
        self.estop_endpoint.force_simple_setup()
        self.estop_keepalive = EstopKeepAlive(self.estop_endpoint)
        self.estop_keepalive.allow()

        # --- clients -------------------------------------------------------
        self.command_client  = self.robot.ensure_client("robot-command")
        self.state_client= self.robot.ensure_client(RobotStateClient.default_service_name)

        # --- spot image -------------------------------------------------
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.gripper_cam_param_client = self.robot.ensure_client(GripperCameraParamClient.default_service_name)
        self.logger = logging.getLogger()
        self.spot_images = SpotImages(
            self.robot,
            self.logger,
            self.image_client,
            self.gripper_cam_param_client
        )

        # power on if needed
        try:
            if not self.robot.is_powered_on():
                print("> Powering on Spot ...")
                self.robot.power_on(timeout_sec=20)
            print("> Robot is powered on.")
        except InternalServerError as e:
            print(f"[!] Error powering on robot: {e}")
            print("Running fault check ...")
            self.check_for_faults()
            print("Reboot the robot and try again.")
            sys.exit(1)
        
        self.dock_id = get_dock_id(self.robot)

        signal.signal(signal.SIGINT, self._clean_shutdown)


    def check_for_faults(self) -> bool:
        """Print current fault state. Returns True if any active faults."""
        state = self.state_client.get_robot_state()

        power_state = getattr(getattr(state, "power_state", None), "motor_power_state", None)
        power_fault = getattr(getattr(state, "power_state", None), "motor_power_state_fault", None)
        print(f"Motor power state: {power_state}")
        if power_fault is not None:
            print(f"Motor power fault flag: {power_fault}")

        system_faults = list(getattr(getattr(state, "system_fault_state", None), "faults", []))
        behavior_faults = list(getattr(getattr(state, "behavior_fault_state", None), "faults", []))

        if not system_faults and not behavior_faults:
            print("No active system or behavior faults.")
            return False

        print(f"Active system faults: {len(system_faults)}")
        for f in system_faults:
            code = getattr(f, "code", "unknown")
            name = getattr(f, "name", "")
            msg = getattr(f, "error_message", "") or getattr(f, "message", "")
            print(f"  SystemFault code={code} name={name} message={msg}")

        print(f"Active behavior faults: {len(behavior_faults)}")
        for f in behavior_faults:
            code = getattr(f, "code", "unknown")
            name = getattr(f, "name", "")
            msg = getattr(f, "error_message", "") or getattr(f, "message", "")
            print(f"  BehaviorFault code={code} name={name} message={msg}")

        return True

    def _clean_shutdown(self, *_):
        print("\n[!] Shutting down VR teleop ...")
        try:
            self.stow_arm()
            blocking_sit(self.command_client, timeout_sec=10)
        finally:
            self.estop_keepalive._end_periodic_check_in()
            self.estop_keepalive.stop()
            sys.exit(0)

    # ─── actuation helpers ───────────────────────────────────────────────
    def _send_arm_pose(self, pos: np.ndarray, quat: np.ndarray, frame_name= BODY_FRAME_NAME):
        """
        pos : (3,) np.ndarray - metres in frame_name
        quat: (4,) np.ndarray - unit quaternion (x,y,z,w)
        frame_name : reference frame for arm pose commands.
        """
        try:
            pose_pb = geometry_pb2.SE3Pose(
                position = geometry_pb2.Vec3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                rotation = geometry_pb2.Quaternion(x=float(quat[0]), y=float(quat[1]),
                                                z=float(quat[2]), w=float(quat[3])))
            arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
                hand_pose = pose_pb,
                frame_name= frame_name,
                seconds   = self.t_exec)
            self.command_client.robot_command(arm_cmd, end_time_secs = time.time() + self.t_exec)
        except Exception as e:
            print(f"[!] Error sending arm pose command: {e}")
    
    def _make_pose_pb(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> geometry_pb2.SE3Pose:
        return geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=float(pos_xyz[0]), y=float(pos_xyz[1]), z=float(pos_xyz[2])),
            rotation=geometry_pb2.Quaternion(x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                                            z=float(quat_xyzw[2]), w=float(quat_xyzw[3]))
    )

    def _ensure_arm_target(self, frame_name: str):
        if self._arm_target_pos is None or self._arm_target_quat is None or self._arm_target_frame != frame_name:
            pos, quat = self.current_ee_pose(frame_name)
            self._arm_target_pos = pos.copy()
            self._arm_target_quat = quat.copy()
            self._arm_target_frame = frame_name

    # ─── public API ──────────────────────────────────────────────────────

    # ─── sensing ───────────
    def current_state(self):
        """Return current robot state as a dictionary."""
        state = self.state_client.get_robot_state()
        return state
        
    def current_ee_pose(self, frame_name= BODY_FRAME_NAME) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos[3], quat[4]) of hand in frame_name."""
        snap = self.current_state().kinematic_state.transforms_snapshot
        pose = get_a_tform_b(snap, frame_name, "hand")
        pos  = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        quat = np.array([pose.rot.x, pose.rot.y, pose.rot.z, pose.rot.w],
                        dtype=np.float32)
        return pos, quat

    def current_ee_pose_se3(self, frame_name= BODY_FRAME_NAME) -> SE3Pose:
        """Return End-effector pose(SE3Pose) in frame_name."""    
        snap = self.current_state().kinematic_state.transforms_snapshot
        pose = get_a_tform_b(snap, frame_name, "hand")
        return pose

    def current_gripper(self) -> float:
        man = self.current_state().manipulator_state
        return float(man.gripper_open_percentage)

    def get_hand_image(self) -> np.ndarray:
        """
        Get the image from the gripper camera.
        Returns
        -------
        np.ndarray
            The image from the gripper camera.
        """

        frame = image_to_cv(self.spot_images.get_hand_rgb_image())

        return frame.copy()
    
    # ─── actuation ─────────
    def send_gripper(self, opening: float):
        opening = opening / 100.0 if opening > 1.0 else opening # convert percentage to fraction if needed
        opening = float(np.clip(opening, 0.0, 1.0))
        grip_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(opening)
        self.command_client.robot_command(grip_cmd)

    def move_arm_to(self,
                  pos_xyz: np.ndarray,
                  quat_xyzw: np.ndarray,
                  gripper: None | float = None,
                  verbose: bool = False,
                  frame_name: str = BODY_FRAME_NAME):
        """
        Absolute command interface (no deltas).

        Parameters
        ----------
        pos_xyz      : (3,) np.ndarray  - metres in frame_name
        quat_xyzw    : (4,) np.ndarray  - unit quaternion
        gripper      : float            - 0.0 closed … 1.0 open
        frame_name   : str              - reference frame for arm pose commands.
        """
        if verbose:
            print(f"Moving arm to: pos=({pos_xyz[0]:.2f}, {pos_xyz[1]:.2f}, {pos_xyz[2]:.2f}), "
              f"quat=({quat_xyzw[0]:.2f}, {quat_xyzw[1]:.2f}, {quat_xyzw[2]:.2f}, {quat_xyzw[3]:.2f})")

         # send arm pose
        self._send_arm_pose(pos_xyz,
                            quat_xyzw,
                            frame_name)
        if gripper is not None:
            self.send_gripper(gripper)
    
    def move_base_with_velocity(self, vx: float, vy: float, wz: float):
        """Send a velocity command to the robot base."""
        try:
            if self.dock_id is None: # If not docked, allow base movement
                vel_cmd = RobotCommandBuilder.synchro_velocity_command(vx, vy, wz)
                self.command_client.robot_command(vel_cmd, end_time_secs= time.time() + self.t_exec)
        except Exception as e:
            print(f"[!] Error sending velocity command: {e}")

    def move_base_to(self, goal_x: float, goal_y: float, goal_heading: float, timeout, frame_name: str = VISION_FRAME_NAME, blocking: bool = False, max_lin_vel: float = 0.6, max_ang_vel: float = 0.8):
        """Send a position command to the robot base with respect to a given frame.(by default, vision frame)"""
        try:
            max_lin_vel = float(max_lin_vel)
            max_ang_vel = float(max_ang_vel)
            if max_lin_vel <= 0.0 or max_ang_vel <= 0.0:
                raise ValueError("max_lin_vel and max_ang_vel must be > 0.")

            speed_limit = geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=max_lin_vel, y=max_lin_vel),
                    angular=max_ang_vel,
                ),
                min_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=-max_lin_vel, y=-max_lin_vel),
                    angular=-max_ang_vel,
                ),
            )
            mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

            pos_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x, goal_y, goal_heading, frame_name, params=mobility_params
            )
            cmd_id = self.command_client.robot_command(pos_cmd, end_time_secs= time.time() + float(timeout))
            if blocking:
                block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=float(timeout))
        except Exception as e:
            print(f"[!] Error sending position command: {e}")

    def move_base_to_pose(self, goal_pose, timeout, frame_name: str = VISION_FRAME_NAME, blocking: bool = False, max_lin_vel: float = 0.6, max_ang_vel: float = 0.8):
        """Send a position command to the robot base with respect to a given frame.(by default, vision frame)
        Parameters
        ----------
        goal_pose : [tx, ty, tz, qx, qy, qz, qw] (list or np.ndarray)
        """
        # convert to SE2Pose
        position = geometry_pb2.Vec2(x=goal_pose[0], y=goal_pose[1])

        # calculate yaw from quaternion
        quat = goal_pose[3:7]
        yaw = math.atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                         1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
        goal_se2 = geometry_pb2.SE2Pose(position=position, angle=yaw)

        
        try:
            max_lin_vel = float(max_lin_vel)
            max_ang_vel = float(max_ang_vel)
            if max_lin_vel <= 0.0 or max_ang_vel <= 0.0:
                raise ValueError("max_lin_vel and max_ang_vel must be > 0.")

            speed_limit = geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=max_lin_vel, y=max_lin_vel),
                    angular=max_ang_vel,
                ),
                min_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=-max_lin_vel, y=-max_lin_vel),
                    angular=-max_ang_vel,
                ),
            )
            mobility_params = spot_command_pb2.MobilityParams(vel_limit=speed_limit)

            pos_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                goal_se2, frame_name, params=mobility_params
            )
            cmd_id = self.command_client.robot_command(pos_cmd, end_time_secs= time.time() + float(timeout))
            if blocking:
                block_for_trajectory_cmd(self.command_client, cmd_id, timeout_sec=float(timeout))
        except Exception as e:
            print(f"[!] Error sending position command: {e}")

    def apply_action(self, action: np.ndarray, verbose=False, frame_name: str = BODY_FRAME_NAME):
        """
        Parameters
        ----------
        action : np.ndarray shape (10,)
            [ Δx Δy Δz , rot6d(6) , gripper_abs ]
        """
        if action.shape != (10,):
            raise ValueError("Action must be np.ndarray with shape (10,)")

        self._ensure_arm_target(frame_name)
        pos = self._arm_target_pos
        quat = self._arm_target_quat

        delta_p   = action[:3]
        rot6d     = action[3:9]
        g_target  = float(action[9])

        # --- pose integration ------------------------------------------
        pos_cmd   = pos + delta_p
        R_curr    = quat_to_matrix(quat)
        R_rel     = rot6d_to_matrix(rot6d)
        R_cmd     = R_curr @ R_rel
        quat_cmd  = matrix_to_quat(R_cmd)

        self._arm_target_pos = pos_cmd
        self._arm_target_quat = quat_cmd
        # --- send to robot ---------------------------------------------
        if verbose:
            print(f"Position({pos_cmd}), Quat({quat_cmd})")
        self.move_arm_to(pos_cmd, quat_cmd, g_target, verbose, frame_name)
        
    def undock(self):
        try:
            self.dock_id = get_dock_id(self.robot)
            if self.dock_id is not None:
                print(f"Robot is docked at {self.dock_id} → undocking …")
                blocking_undock(self.robot, timeout=20)
                self.dock_id = get_dock_id(self.robot)
                print("Robot undocked.")
                return True
            else:
                print("Robot is not docked.")
                return False
        except Exception as e:
            print(f"[!] Error during undocking: {e}")
            return False
    def dock(self):
        try:
            self.dock_id = get_dock_id(self.robot)
            if self.dock_id is None:
                self.send_gripper(0.0)  # Close gripper before docking
                print("Robot is undocked → docking ...")
                # Stand before trying to dock.
                blocking_stand(self.command_client, timeout_sec=10)
                blocking_dock_robot(self.robot, dock_id=520, timeout=30)
                self.dock_id = get_dock_id(self.robot)
                print(f"Robot docked at id={self.dock_id} and powered off.")
                print("> Powering on again...")
                self.robot.power_on(timeout_sec=20)
                print("> Robot is powered on.")
                return True
            else:
                print(f"Robot is already docked at {self.dock_id}.")
                return False
        except Exception as e:
            print(f"[!] Error during docking: {e}")
            return False
    
    def dock_undock(self):
        try:
            self.dock_id = get_dock_id(self.robot)
            if self.dock_id is not None:
                self.undock()
            else:
                self.dock()
        except Exception as e:
            print(f"[!] Error during docking/undocking: {e}")
    
    def stow_arm(self):
        """Stow the arm to a safe position."""
        print("Stowing arm...")
        try:
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            cmd_id = self.command_client.robot_command(stow_cmd, end_time_secs=time.time() + 5.0)
            block_until_arm_arrives(self.command_client, cmd_id)
            self.stowed = True
            print("Arm stowed.")
        except Exception as e:
            print(f"[!] Error stowing arm: {e}")

    def unstow_arm(self):
        """Unstow the arm to a ready position."""
        print("Unstowing arm...")
        try:
            unstow_cmd = RobotCommandBuilder.arm_ready_command()
            cmd_id = self.command_client.robot_command(unstow_cmd, end_time_secs=time.time() + 5.0)
            block_until_arm_arrives(self.command_client, cmd_id)
            self.stowed = False
            print("Arm unstowed.")
        except Exception as e:
            print(f"[!] Error unstowing arm: {e}")
    
    def stand(self):
        """Stand the robot up."""
        print("Standing up...")

        # Checking if the robot is powered on
        if not self.robot.is_powered_on():
            print("> Please power on the robot first.")
            return
        try:
            blocking_stand(self.command_client, timeout_sec=10)
        except Exception as e:
            print(f"[!] Error standing up: {e}")

    def sit(self):
        """Sit the robot down."""
        print("Sitting down...")

        # Checking if the robot is powered on
        if not self.robot.is_powered_on():
            print("> Please power on the robot first.")
            return
        try:
            blocking_sit(self.command_client, timeout_sec=10)
        except Exception as e:
            print(f"[!] Error sitting down: {e}")

    def reset_pose(self, pose=[0.9, 0, 0.2, 0, 0.7071068, 0, 0.7071068], frane_name= BODY_FRAME_NAME):
        """Move the arm to a default pose.
        Parameters
        ----------
        pose : list of float, shape (7,)
            [x, y, z, qx, qy, qz, qw]
        """
        # parameter validation
        if len(pose) != 7:
            raise ValueError("Pose must be a list of 7 elements: [x, y, z, qx, qy, qz, qw]")
        if not np.isclose(np.linalg.norm(pose[3:]), 1.0):
            raise ValueError("Quaternion must be a unit quaternion.")
        
        cmd_id = self.send_arm_cartesian_hybrid(
                    pos_xyz=np.array(pose[:3]),
                    quat_xyzw=np.array(pose[3:]),
                    seconds=2.0,
                    max_lin_vel=0.25,
                    max_ang_vel=0.8,
                    root_frame=frane_name,  # or "vision"
                )
        try:
            block_until_arm_arrives(self.command_client, cmd_id)
        except Exception as e:
            print(f"[!] Error waiting for reset pose: {e}")
        try:
            pos, quat = self.current_ee_pose(frane_name)
            self._arm_target_pos = pos.copy()
            self._arm_target_quat = quat.copy()
            self._arm_target_frame = frane_name
        except Exception as e:
            print(f"[!] Error syncing target pose after reset: {e}")

    def _duration_from_seconds(self, sec: float) -> duration_pb2.Duration:
        """Convert float seconds to protobuf Duration (seconds + nanos)."""
        sec = max(0.0, float(sec))
        whole = int(math.floor(sec))
        nanos = int(round((sec - whole) * 1e9))
        if nanos >= 1_000_000_000:  # handle rounding edge cases
            whole += 1
            nanos = 0
        d = duration_pb2.Duration()
        d.seconds = whole
        d.nanos = nanos
        return d
    def send_arm_cartesian_hybrid(
        self,
        pos_xyz, quat_xyzw,
        *,
        seconds: float = 0.25,
        max_lin_vel: float = 0.25,
        max_ang_vel: float = 0.8,
        max_accel: float = 2.0,
        root_frame: str = "body",
        task_T_root: geometry_pb2.SE3Pose | None = None,
        desired_wrench_in_task: tuple[float, float, float, float, float, float] | None = None
    ):
        # Build ArmCartesianCommand
        cart_req = arm_command_pb2.ArmCartesianCommand.Request()
        cart_req.root_frame_name = root_frame
        if task_T_root is not None:
            cart_req.root_tform_task.CopyFrom(task_T_root)

        pose_pb = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=float(pos_xyz[0]), y=float(pos_xyz[1]), z=float(pos_xyz[2])),
            rotation=geometry_pb2.Quaternion(x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
                                            z=float(quat_xyzw[2]), w=float(quat_xyzw[3]))
        )

        pose_pt = trajectory_pb2.SE3TrajectoryPoint(
            pose=pose_pb,
            time_since_reference=self._duration_from_seconds(seconds)
        )
        cart_req.pose_trajectory_in_task.points.append(pose_pt)

        if desired_wrench_in_task is not None:
            Fx, Fy, Fz, Tx, Ty, Tz = desired_wrench_in_task
            wrench_pt = arm_command_pb2.WrenchTrajectoryPoint(
                wrench=geometry_pb2.Wrench(
                    force=geometry_pb2.Vec3(x=float(Fx), y=float(Fy), z=float(Fz)),
                    torque=geometry_pb2.Vec3(x=float(Tx), y=float(Ty), z=float(Tz)),
                ),
                time_since_reference=self._duration_from_seconds(seconds)
            )
            cart_req.wrench_trajectory_in_task.points.append(wrench_pt)

        cart_req.max_linear_velocity.value  = float(max_lin_vel)
        cart_req.max_angular_velocity.value = float(max_ang_vel)
        cart_req.maximum_acceleration.value = float(max_accel)

        arm_req  = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=cart_req)
        sync_req = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm_req)
        cmd = robot_command_pb2.RobotCommand(synchronized_command=sync_req)

        return self.command_client.robot_command(cmd, end_time_secs=time.time() + seconds)




