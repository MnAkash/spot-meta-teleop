#!/usr/bin/env python3
"""
Natural tele-operation of Boston Dynamics Spot and its arm with Meta Quest controllers, spacemouse and keyboard.

Teleop modes:
- meta: Meta Quest controller + keyboard A/B/X/Y actions
- spacemouse: SpaceMouse + keyboard A/B/X/Y actions
- keyboard: keyboard base teleop + A/B/X/Y actions (no arm control)

The script assumes you already have:
    - Spot SDK 5.0.0 installed in the active conda env
    - A working meta quest setup with adb enabled
    - The `reader.py` module in the same directory, which reads the Meta Quest controller

Usage example:
    python teleop_spot.py --teleop-type meta --use-depth --force-limit-disable
    python teleop_spot.py --teleop-type spacemouse
    python teleop_spot.py --teleop-type keyboard


Author: Moniruzzaman Akash
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict

import numpy as np

from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import Quat

from spot_teleop.camera_streamer import CameraStreamer
from spot_teleop.demo_recorder import DemoRecorder
from spot_teleop.reader import get_connecteed_device_ip
from spot_teleop.spot_controller import SpotRobotController
from spot_teleop.utils.spot_utils import map_controller_to_robot, mat_to_se3
from spot_teleop.utils.teleop_inputs import KeyboardInputHelper, MetaInputHelper, SpaceMouseInputHelper

import logging


class SpotVRTeleop:
    MAX_VEL_X = 0.6          # [m/s] forward/back
    MAX_VEL_Y = 0.6          # [m/s] left/right
    MAX_YAW = 0.8            # [rad/s] spin

    ARM_SCALE_META = 2.0
    ARM_SCALE_SPACEMOUSE = 0.02
    ARM_ROT_SCALE_SPACEMOUSE = 0.08

    VEL_SMOOTH_ALPHA = 0.35
    ARM_SMOOTH_ALPHA = 0.8
    DEFAULT_POSE = [0.7, 0, 0.4, 0, 0, 0, 1]

    def __init__(
        self,
        robot_ip,
        username,
        password,
        *,
        teleop_type: str = "meta",
        force_limit_enable: bool = True,
        use_depth: bool = False,
        home_pose=None,
        meta_quest_ip=None,
        demo_image_preview=True,
    ):
        self.teleop_type = str(teleop_type).lower()
        if self.teleop_type not in {"meta", "keyboard", "spacemouse"}:
            raise ValueError(f"Unsupported teleop_type: {teleop_type}")

        self.spot = SpotRobotController(robot_ip, username, password)
        self.logger = logging.getLogger("spot-teleop")
        self.home_pose = self.DEFAULT_POSE if home_pose is None else home_pose
        self.demo_image_preview = demo_image_preview
        self.external_camera = None
        try:
            self.external_camera = CameraStreamer(width=320, height=240, fps=30)
        except Exception as e:
            self.logger.warning("External camera disabled: %s", e)

        self.recorder = DemoRecorder(
            robot=self.spot.robot,
            spot_images=self.spot.spot_images,
            state_client=self.spot.state_client,
            external_camera=self.external_camera,
            use_depth=use_depth,
            image_size=(320, 240),
            out_dir="demos",
            fps=10,
            preview=self.demo_image_preview,
        )

        # Runtime states
        self.arm_anchor_ctrl = None
        self.arm_anchor_robot = None
        self.prev_r_grip = False
        self.first_frame_arm = True
        self.prev_goal_xyz = None

        self._vx_f = 0.0
        self._vy_f = 0.0
        self._wz_f = 0.0
        self._vqx_f = 0.0
        self._vqy_f = 0.0
        self._vqz_f = 0.0
        self._vqw_f = 1.0
        self._Ff = np.zeros(3, dtype=float)

        self.force_limit_enabled = bool(force_limit_enable)
        self._prev_meta_abxy = {"a": False, "b": False, "x": False, "y": False}
        self._kb_gripper_open = True

        # Input helpers
        self.keyboard_input: KeyboardInputHelper | None = KeyboardInputHelper(
            enable_base_keys=(self.teleop_type == "keyboard")
        )
        try:
            self.keyboard_input.start()
        except Exception as e:
            if self.teleop_type == "keyboard":
                raise RuntimeError(f"Keyboard teleop requires pynput. {e}") from e
            self.logger.warning("Keyboard shortcuts disabled (%s).", e)
            self.keyboard_input = None

        self.meta_input: MetaInputHelper | None = None
        self.spacemouse_input: SpaceMouseInputHelper | None = None
        if self.teleop_type == "meta":
            self.meta_input = MetaInputHelper(meta_quest_ip)
        elif self.teleop_type == "spacemouse":
            self.spacemouse_input = SpaceMouseInputHelper()

    # ---------------- utility ---------------- #
    def close(self):
        if self.spacemouse_input is not None:
            try:
                self.spacemouse_input.stop()
            except Exception:
                pass
        if self.keyboard_input is not None:
            try:
                self.keyboard_input.stop()
            except Exception:
                pass

    def _smooth(self, prev: float, target: float, alpha: float) -> float:
        return (1 - alpha) * prev + alpha * target

    def _button_pressed(self, buttons: Dict, *names: str, threshold: float = 0.5) -> bool:
        for name in names:
            if name not in buttons:
                continue
            v = buttons.get(name)
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v) > threshold
            if isinstance(v, (tuple, list, np.ndarray)) and len(v) > 0:
                try:
                    return float(v[0]) > threshold
                except Exception:
                    return bool(v[0])
            return bool(v)
        return False

    def _axis1(self, buttons: Dict, name: str, default: float = 0.0) -> float:
        v = buttons.get(name, (default,))
        if isinstance(v, (tuple, list, np.ndarray)):
            if len(v) == 0:
                return float(default)
            try:
                return float(v[0])
            except Exception:
                return float(default)
        if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)):
            return float(v)
        return float(default)

    def _axis2(self, buttons: Dict, name: str) -> tuple[float, float]:
        v = buttons.get(name, (0.0, 0.0))
        if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except Exception:
                return 0.0, 0.0
        return 0.0, 0.0

    def _consume_action_key(self, name: str) -> bool:
        if self.keyboard_input is None:
            return False
        return self.keyboard_input.consume_action(name)

    def _apply_common_actions(self, trig_a: bool, trig_b: bool, trig_x: bool, trig_y: bool):
        if trig_a:
            self.spot.dock_undock()
        if trig_b:
            self.toggle_recording()
        if trig_x:
            self.spot.stow_arm()
        if trig_y:
            self.spot.reset_pose(pose=self.home_pose)

    def _rpy_to_quat(self, roll: float, pitch: float, yaw: float) -> Quat:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return Quat(w, x, y, z)

    def _apply_force_limit(
        self,
        delta_cmd: np.ndarray,
        pose_now,
        soft_limits: np.ndarray,
        hard_limits: np.ndarray,
        manipulator_state=None,
    ) -> np.ndarray:
        """Limit translational delta based on contact force."""
        parallel_min_scale = 0.0
        f_detect = 0.8
        alpha_f = 0.2

        if np.linalg.norm(delta_cmd) < 1e-9:
            return delta_cmd

        if manipulator_state is None:
            manipulator_state = self.spot.state_client.get_robot_state().manipulator_state

        fH = manipulator_state.estimated_end_effector_force_in_hand
        f_hand = np.array([fH.x, fH.y, fH.z], dtype=float)

        try:
            R_BH = pose_now.rot.to_matrix()
        except AttributeError:
            qx, qy, qz, qw = pose_now.rot.x, pose_now.rot.y, pose_now.rot.z, pose_now.rot.w
            R_BH = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ], dtype=float)

        F_body = R_BH @ f_hand
        self._Ff = (1.0 - alpha_f) * self._Ff + alpha_f * F_body
        F_body = self._Ff
        normF = float(np.linalg.norm(F_body))

        delta_out = delta_cmd
        if normF > f_detect:
            n_c = -F_body / (normF + 1e-9)
            d_par_mag = float(np.dot(delta_cmd, n_c))
            d_par = d_par_mag * n_c
            d_ortho = delta_cmd - d_par

            if d_par_mag > 0.0:
                abs_nc = np.abs(n_c) + 1e-9
                soft_eff = float(np.min(soft_limits / abs_nc))
                hard_eff = float(np.min(hard_limits / abs_nc))
                if normF <= soft_eff:
                    alpha = 1.0
                elif normF >= hard_eff:
                    alpha = parallel_min_scale
                else:
                    k = (normF - soft_eff) / (hard_eff - soft_eff + 1e-9)
                    alpha = (1.0 - k) * (1.0 - parallel_min_scale) + parallel_min_scale
                delta_out = d_ortho + alpha * d_par
        return delta_out

    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.recorder.start()
        else:
            self.recorder.stop()
            self.spot.reset_pose(pose=self.home_pose)

    # ---------------- run dispatch ---------------- #
    def run(self):
        try:
            if self.teleop_type == "meta":
                self._run_meta()
            elif self.teleop_type == "spacemouse":
                self._run_spacemouse()
            else:
                self._run_keyboard()
        finally:
            self.close()

    def _run_meta(self):
        if self.meta_input is None:
            raise RuntimeError("Meta teleop selected but Meta input helper not initialized.")

        rate_hz = 30.0
        dt = 1.0 / rate_hz
        t_prev = time.time()
        soft_limits = np.array([5.0, 5.0, 5.0], dtype=float)
        hard_limits = np.array([8.0, 8.0, 8.0], dtype=float)

        while True:
            poses, buttons = self.meta_input.get()
            if len(poses) == 0 or len(buttons) == 0:
                print("[!] No poses or buttons received from Meta Quest.")
                time.sleep(dt)
                continue

            key_a = self._consume_action_key("a")
            key_b = self._consume_action_key("b")
            key_x = self._consume_action_key("x")
            key_y = self._consume_action_key("y")

            meta_a = self._button_pressed(buttons, "A", "a")
            meta_b = self._button_pressed(buttons, "B", "b")
            meta_x = self._button_pressed(buttons, "X", "x")
            meta_y = self._button_pressed(buttons, "Y", "y")

            trig_a = key_a or (meta_a and not self._prev_meta_abxy["a"])
            trig_b = key_b or (meta_b and not self._prev_meta_abxy["b"])
            trig_x = key_x or (meta_x and not self._prev_meta_abxy["x"])
            trig_y = key_y or (meta_y and not self._prev_meta_abxy["y"])
            self._apply_common_actions(trig_a, trig_b, trig_x, trig_y)
            self._prev_meta_abxy["a"] = meta_a
            self._prev_meta_abxy["b"] = meta_b
            self._prev_meta_abxy["x"] = meta_x
            self._prev_meta_abxy["y"] = meta_y

            try:
                ljx, ljy = self._axis2(buttons, "leftJS")
                rjx, _ = self._axis2(buttons, "rightJS")
                vx = self.MAX_VEL_X * ljy
                vy = self.MAX_VEL_Y * (-ljx)
                wz = self.MAX_YAW * (-rjx)
                self._vx_f = self._smooth(self._vx_f, vx, self.VEL_SMOOTH_ALPHA)
                self._vy_f = self._smooth(self._vy_f, vy, self.VEL_SMOOTH_ALPHA)
                self._wz_f = self._smooth(self._wz_f, wz, self.VEL_SMOOTH_ALPHA)
                self.spot.move_base_with_velocity(self._vx_f, self._vy_f, self._wz_f)
            except Exception as e:
                print(f"[!] Error sending base command: {e}")

            trigger_val = 1.0 - self._axis1(buttons, "rightTrig", default=0.0)
            self.spot.send_gripper(trigger_val)

            try:
                lgrip = self._axis1(buttons, "leftGrip", default=0.0) > 0.5 or self._button_pressed(
                    buttons, "LG"
                )
                rmat_raw = poses["r"]
                rmat = map_controller_to_robot(rmat_raw)

                if lgrip and not self.prev_r_grip:
                    self.arm_anchor_ctrl = rmat.copy()
                    self.arm_anchor_robot = self.spot.current_ee_pose_se3()

                if lgrip:
                    delta = np.linalg.inv(self.arm_anchor_ctrl) @ rmat
                    delta_scaled = delta.copy()
                    delta_scaled[0:3, 3] *= self.ARM_SCALE_META
                    delta_pose = mat_to_se3(delta_scaled)
                    goal = self.arm_anchor_robot * delta_pose
                    desired_xyz = np.array([goal.x, goal.y, goal.z], dtype=float)

                    pose_now = self.spot.current_ee_pose_se3()
                    x_now = np.array([pose_now.x, pose_now.y, pose_now.z], dtype=float)
                    delta_cmd = desired_xyz - x_now
                    if self.force_limit_enabled:
                        delta_out = self._apply_force_limit(delta_cmd, pose_now, soft_limits, hard_limits)
                    else:
                        delta_out = delta_cmd
                    blended_xyz = x_now + delta_out

                    cmd_pos = np.array(blended_xyz)
                    cmd_quat = np.array([goal.rot.x, goal.rot.y, goal.rot.z, goal.rot.w])
                    self.spot.send_arm_cartesian_hybrid(
                        pos_xyz=cmd_pos,
                        quat_xyzw=cmd_quat,
                        seconds=0.25,
                        max_lin_vel=0.35,
                        max_ang_vel=1.5,
                        root_frame="body",
                    )
                    self.prev_goal_xyz = blended_xyz.copy()

                self.prev_r_grip = lgrip
            except Exception as e:
                print(f"[!] Error sending arm command: {e}")

            sleep = max(0.0, dt - (time.time() - t_prev))
            time.sleep(sleep)
            t_prev = time.time()
            if self.demo_image_preview:
                self.recorder.poll_preview()

    def _run_spacemouse(self):
        if self.spacemouse_input is None:
            raise RuntimeError("SpaceMouse teleop selected but SpaceMouse helper not initialized.")

        rate_hz = 30.0
        dt = 1.0 / rate_hz
        t_prev = time.time()
        soft_limits = np.array([5.0, 5.0, 5.0], dtype=float)
        hard_limits = np.array([8.0, 8.0, 8.0], dtype=float)
        toggle_movement = True  # True = base mode, False = arm mode

        while True:
            axes, sm_buttons = self.spacemouse_input.get_axes_buttons()

            trig_a = self._consume_action_key("a")
            trig_b = self._consume_action_key("b")
            trig_x = self._consume_action_key("x")
            trig_y = self._consume_action_key("y")

            if trig_a:
                self.spot.dock_undock()
            if trig_b:
                self.toggle_recording()
            if trig_x:
                self.spot.stow_arm()
                toggle_movement = True
            if trig_y:
                self.spot.reset_pose(pose=self.home_pose)
                toggle_movement = False
                self.first_frame_arm = True

            if toggle_movement:
                self.first_frame_arm = True
                try:
                    vx = self.MAX_VEL_X * axes["y"]
                    vy = self.MAX_VEL_Y * (-axes["x"])
                    wz = self.MAX_YAW * (-axes["yaw"])
                    self._vx_f = self._smooth(self._vx_f, vx, self.VEL_SMOOTH_ALPHA)
                    self._vy_f = self._smooth(self._vy_f, vy, self.VEL_SMOOTH_ALPHA)
                    self._wz_f = self._smooth(self._wz_f, wz, self.VEL_SMOOTH_ALPHA)
                    self.spot.move_base_with_velocity(self._vx_f, self._vy_f, self._wz_f)
                except Exception as e:
                    print(f"[!] Error sending base command: {e}")
            else:
                try:
                    trigger_val = 1 - sm_buttons[1] if len(sm_buttons) > 1 else 1
                    self.spot.send_gripper(trigger_val)

                    move_commanded = any(abs(axes[k]) > 0.0 for k in ["x", "y", "z", "roll", "pitch", "yaw"])
                    if not move_commanded:
                        sleep = max(0.0, dt - (time.time() - t_prev))
                        time.sleep(sleep)
                        t_prev = time.time()
                        if self.demo_image_preview:
                            self.recorder.poll_preview()
                        continue

                    if self.first_frame_arm:
                        self.first_frame_arm = False
                        self.arm_anchor_robot = self.spot.current_ee_pose_se3()
                        self.prev_goal_xyz = None

                    robot_state = self.spot.state_client.get_robot_state()
                    snap = robot_state.kinematic_state.transforms_snapshot
                    pose_now = get_a_tform_b(snap, BODY_FRAME_NAME, "hand")
                    x_now = np.array([pose_now.x, pose_now.y, pose_now.z], dtype=float)

                    delta_cmd = np.array(
                        [axes["y"], -axes["x"], axes["z"]],
                        dtype=float,
                    ) * self.ARM_SCALE_SPACEMOUSE

                    if self.force_limit_enabled:
                        delta_out = self._apply_force_limit(
                            delta_cmd,
                            pose_now,
                            soft_limits,
                            hard_limits,
                            manipulator_state=robot_state.manipulator_state,
                        )
                    else:
                        delta_out = delta_cmd
                    blended_xyz = x_now + delta_out

                    d_roll = axes["roll"] * self.ARM_ROT_SCALE_SPACEMOUSE
                    d_pitch = axes["pitch"] * self.ARM_ROT_SCALE_SPACEMOUSE
                    d_yaw = -axes["yaw"] * self.ARM_ROT_SCALE_SPACEMOUSE
                    delta_quat = self._rpy_to_quat(d_roll, d_pitch, d_yaw)
                    goal_rot = delta_quat * pose_now.rot

                    self._vqx_f = self._smooth(self._vqx_f, goal_rot.x, self.ARM_SMOOTH_ALPHA)
                    self._vqy_f = self._smooth(self._vqy_f, goal_rot.y, self.ARM_SMOOTH_ALPHA)
                    self._vqz_f = self._smooth(self._vqz_f, goal_rot.z, self.ARM_SMOOTH_ALPHA)
                    self._vqw_f = self._smooth(self._vqw_f, goal_rot.w, self.ARM_SMOOTH_ALPHA)
                    quat_xyzw = np.array([self._vqx_f, self._vqy_f, self._vqz_f, self._vqw_f], dtype=float)

                    self.spot.send_arm_cartesian_hybrid(
                        pos_xyz=blended_xyz,
                        quat_xyzw=quat_xyzw,
                        seconds=max(0.02, dt),
                        max_lin_vel=0.25,
                        max_ang_vel=1.5,
                        root_frame="body",
                    )
                    self.prev_goal_xyz = blended_xyz.copy()
                except Exception as e:
                    print(f"[!] Error sending arm command: {e}")

            sleep = max(0.0, dt - (time.time() - t_prev))
            time.sleep(sleep)
            t_prev = time.time()
            if self.demo_image_preview:
                self.recorder.poll_preview()

    def _run_keyboard(self):
        rate_hz = 30.0
        dt = 1.0 / rate_hz
        t_prev = time.time()
        try:
            # Keyboard mode starts with gripper open.
            self.spot.send_gripper(1.0)
        except Exception as e:
            print(f"[!] Error setting initial gripper state: {e}")

        while True:
            trig_a = self._consume_action_key("a")
            trig_b = self._consume_action_key("b")
            trig_x = self._consume_action_key("x")
            trig_y = self._consume_action_key("y")
            self._apply_common_actions(trig_a, trig_b, trig_x, trig_y)

            toggle_gripper = False
            if self.keyboard_input is not None:
                toggle_gripper = self.keyboard_input.consume_once("space")
            if toggle_gripper:
                self._kb_gripper_open = not self._kb_gripper_open
                try:
                    self.spot.send_gripper(1.0 if self._kb_gripper_open else 0.0)
                except Exception as e:
                    print(f"[!] Error toggling gripper: {e}")

            forward, right, yaw_right = (0.0, 0.0, 0.0)
            if self.keyboard_input is not None:
                forward, right, yaw_right = self.keyboard_input.get_base_motion()

            vx = self.MAX_VEL_X * forward
            vy = self.MAX_VEL_Y * (-right)   # right key -> move right
            wz = self.MAX_YAW * (-yaw_right) # o key -> rotate right
            self._vx_f = self._smooth(self._vx_f, vx, self.VEL_SMOOTH_ALPHA)
            self._vy_f = self._smooth(self._vy_f, vy, self.VEL_SMOOTH_ALPHA)
            self._wz_f = self._smooth(self._wz_f, wz, self.VEL_SMOOTH_ALPHA)

            try:
                self.spot.move_base_with_velocity(self._vx_f, self._vy_f, self._wz_f)
            except Exception as e:
                print(f"[!] Error sending base command: {e}")

            sleep = max(0.0, dt - (time.time() - t_prev))
            time.sleep(sleep)
            t_prev = time.time()
            if self.demo_image_preview:
                self.recorder.poll_preview()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teleop-type", choices=["meta", "keyboard", "spacemouse"], default="meta")
    parser.add_argument("--meta-quest-ip", default=os.environ.get("META_QUEST_IP", "192.168.1.35"))
    parser.add_argument("--force-limit-disable",action="store_true", help="Disable force-aware arm limiting (enabled by default).",)
    parser.add_argument("--use-depth", action="store_true", help="Record camera depth streams (disabled by default).")
    parser.add_argument("--demo-image-preview", action="store_true")
    args = parser.parse_args()

    robot_ip = os.environ.get("SPOT_ROBOT_IP", "192.168.1.138")
    user = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
    password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")

    print(f"Connecting to Spot at {robot_ip} ...")
    print(f"user: {user}, password: {len(password) * '*'}")

    meta_ip = args.meta_quest_ip
    if args.teleop_type == "meta" and (meta_ip is None or str(meta_ip).strip() == ""):
        meta_ip = get_connecteed_device_ip()
        if meta_ip is None:
            print("[!] Could not find connected Meta Quest device IP via ADB.")
            print("    Please ensure ADB is set up and allowed access.")
            return

    home_pose = [0.55, 0.0, 0.55, 0.0, 0.5, 0, 0.8660254]
    # home_pose = [0.6328, 0.0054, 0.3568, -0.7006, -0.1321, -0.018, 0.701]
    teleop = SpotVRTeleop(
        robot_ip,
        user,
        password,
        teleop_type=args.teleop_type,
        force_limit_enable=not args.force_limit_disable,
        use_depth=args.use_depth,
        home_pose=home_pose,
        meta_quest_ip=meta_ip,
        demo_image_preview=args.demo_image_preview,
    )
    teleop.run()

if __name__ == "__main__":
    main()
