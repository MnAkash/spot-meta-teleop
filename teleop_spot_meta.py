#!/usr/bin/env python3
"""
=================
Natural tele-operation of Boston Dynamics Spot and its arm with Meta Quest controllers.

The script assumes you already have:
    - Spot SDK 5.0.0 installed in the active conda env
    - A working meta quest setup with adb enabled
    - The `reader.py` module in the same directory, which reads the Meta Quest controller

Author: Moniruzzaman Akash
"""
import argparse, math, signal, sys, time, os
import numpy as np
from typing import Tuple, Dict
import threading

from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME
from spot_teleop.spot_controller import SpotRobotController

from spot_teleop.reader import OculusReader, get_connecteed_device_ip
from spot_teleop.demo_recorder import DemoRecorder
from spot_teleop.utils.spot_utils import mat_to_se3, map_controller_to_robot
import logging
try:
    from pynput import keyboard
except Exception:
    keyboard = None

class SpotVRTeleop:
    MAX_VEL_X = 0.6          # [m/s] forward/back
    MAX_VEL_Y = 0.6          # [m/s] left/right
    MAX_YAW   = 0.8          # [rad/s] spin
    ARM_SCALE = 2.0          # [m per m] controller-to-arm translation
    DEFAULT_POSE = [0.7, 0, 0.4, 0, 0, 0, 1] # x,y,z, qx,qy,qz,qw

    VEL_SMOOTH_ALPHA = 0.35  # simple first-order smoothing for base


    def __init__(self, robot_ip, username, password, home_pose=None, meta_quest_ip=None, demo_image_preview=True):
        self.oculus_reader = OculusReader(ip_address=meta_quest_ip)
        self.spot = SpotRobotController(robot_ip, username, password)
        self.logger = logging.getLogger("vr-teleop")

        # ---- runtime vars ---
        self.arm_anchor_ctrl  = None   # 4×4 SE(3) when grip first pressed
        self.arm_anchor_robot = None   # SE3Pose   ^ … corresponding robot pose
        self.prev_r_grip      = False
        self.base_enabled     = False

        if home_pose == None:
            self.home_pose = self.DEFAULT_POSE
        else:
            self.home_pose = home_pose

        # Base smoothing state
        self._vx_f = 0.0
        self._vy_f = 0.0
        self._wz_f = 0.0

        self.stowed = False  # arm stowed at start
        self.demo_image_preview = demo_image_preview  # show camera preview in demo mode1

        self.recorder = DemoRecorder(
            robot=self.spot.robot,
            spot_images=self.spot.spot_images,
            state_client=self.spot.state_client,
            image_size = (320, 240),  # maintaining 4:3 aspect ratio
            out_dir="demos",
            fps=10,
            preview=self.demo_image_preview)
        self._kb_once = set()
        self._kb_held = set()
        self._kb_lock = threading.Lock()
        self._kb_listener = None
        self._start_keyboard_listener()
        self._prev_meta_abxy = {"a": False, "b": False, "x": False, "y": False}
        self.force_limit_enabled = False
        self._Ff = np.zeros(3, dtype=float)

    def _start_keyboard_listener(self):
        if keyboard is None:
            self.logger.warning("pynput not available; keyboard A/B/X/Y shortcuts disabled.")
            return
        self._kb_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self._kb_listener.daemon = True
        self._kb_listener.start()

    def _on_key_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            return
        if k not in {"a", "b", "x", "y"}:
            return
        with self._kb_lock:
            if k not in self._kb_held:
                self._kb_held.add(k)
                self._kb_once.add(k)

    def _on_key_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            return
        with self._kb_lock:
            self._kb_held.discard(k)

    def _consume_key_once(self, key_name: str) -> bool:
        with self._kb_lock:
            if key_name in self._kb_once:
                self._kb_once.remove(key_name)
                return True
        return False

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

    def get_meta_quest(self):
        return self.oculus_reader.get_transformations_and_buttons()

    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.recorder.start()
        else:
            self.recorder.stop()
            self.spot.reset_pose(pose=self.home_pose) # x,y,z, qx,qy,qz,qw

    def _smooth(self, prev: float, target: float, alpha: float) -> float:
        """1st order filter."""
        return (1 - alpha) * prev + alpha * target

    def _apply_force_limit(self, delta_cmd: np.ndarray, pose_now, soft_limits: np.ndarray, hard_limits: np.ndarray) -> np.ndarray:
        """Limit commanded translational delta based on end-effector force."""
        PARALLEL_MIN_SCALE = 0.0   # residual motion allowed along normal at/above hard (0..1)
        F_DETECT = 0.8             # N; ignore tiny forces to avoid flicker
        ALPHA_F = 0.2              # force EWMA smoothing factor

        if np.linalg.norm(delta_cmd) < 1e-9:
            return delta_cmd

        man = self.spot.state_client.get_robot_state().manipulator_state
        fH = man.estimated_end_effector_force_in_hand
        f_hand = np.array([fH.x, fH.y, fH.z], dtype=float)

        # BODY <- HAND rotation from CURRENT EE pose
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

        # smooth force a bit to avoid chatter
        self._Ff = (1.0 - ALPHA_F) * self._Ff + ALPHA_F * F_body
        F_body = self._Ff
        normF = float(np.linalg.norm(F_body))

        delta_out = delta_cmd
        if normF > F_DETECT:
            # contact normal to LIMIT ALONG = INTO the surface = -unit(force)
            n_c = -F_body / (normF + 1e-9)

            # decompose commanded delta w.r.t. contact normal
            d_par_mag = float(np.dot(delta_cmd, n_c))  # >0 = moving INTO contact
            d_par = d_par_mag * n_c
            d_ortho = delta_cmd - d_par

            if d_par_mag > 0.0:
                # effective thresholds along n_c (tightest axis wins)
                abs_nc = np.abs(n_c) + 1e-9
                soft_eff = float(np.min(soft_limits / abs_nc))
                hard_eff = float(np.min(hard_limits / abs_nc))

                # scale alpha(F): 1 -> PARALLEL_MIN_SCALE as force rises from soft->hard
                if normF <= soft_eff:
                    alpha = 1.0
                elif normF >= hard_eff:
                    alpha = PARALLEL_MIN_SCALE
                else:
                    k = (normF - soft_eff) / (hard_eff - soft_eff + 1e-9)
                    alpha = (1.0 - k) * (1.0 - PARALLEL_MIN_SCALE) + PARALLEL_MIN_SCALE

                delta_out = d_ortho + alpha * d_par
            else:
                delta_out = delta_cmd  # moving away or tangential -> don't restrict

        return delta_out
    
    def run(self):
        rate_hz = 30.0
        dt = 1.0 / rate_hz
        t_prev = time.time()
        
        # --- per-axis limits (N): start scaling at soft, stop by hard ---
        soft_limits = np.array([5.0, 5.0, 5.0], dtype=float)   # Fx, Fy, Fz
        hard_limits = np.array([8.0, 8.0, 8.0], dtype=float)
        self.prev_goal_xyz = None

        t = 0
        while True:
            poses, buttons = self.get_meta_quest()

            if len(poses) == 0 or len(buttons) == 0:
                print("[!] No poses or buttons received from Meta Quest.")
                time.sleep(dt)
                continue

            key_a = self._consume_key_once("a")
            key_b = self._consume_key_once("b")
            key_x = self._consume_key_once("x")
            key_y = self._consume_key_once("y")

            meta_a = self._button_pressed(buttons, "A", "a")
            meta_b = self._button_pressed(buttons, "B", "b")
            meta_x = self._button_pressed(buttons, "X", "x")
            meta_y = self._button_pressed(buttons, "Y", "y")

            trig_a = key_a or (meta_a and not self._prev_meta_abxy["a"])
            trig_b = key_b or (meta_b and not self._prev_meta_abxy["b"])
            trig_x = key_x or (meta_x and not self._prev_meta_abxy["x"])
            trig_y = key_y or (meta_y and not self._prev_meta_abxy["y"])

            # ---------------- POWER TOGGLE (A/B) ------------------
            if trig_a:
                self.spot.dock_undock()
            if trig_b:
                self.toggle_recording()
            # ------------------ STOW/UNSTOW ARM -------------------
            if trig_x:
                self.spot.stow_arm()
            if trig_y:
                # self.spot.unstow_arm()
                # Move arm to a ready position
                self.spot.reset_pose(pose=self.home_pose) # x,y,z, qx,qy,qz,qw

            self._prev_meta_abxy["a"] = meta_a
            self._prev_meta_abxy["b"] = meta_b
            self._prev_meta_abxy["x"] = meta_x
            self._prev_meta_abxy["y"] = meta_y

            # ---------------- BASE  (LEFT HAND) -------------------
            lgrip  = buttons.get('leftGrip', (0.0,))[0] > 0.5 or buttons.get('LG', False)
            ljx, ljy = buttons.get('leftJS', (0.0, 0.0))
            rjx, rjy = buttons.get('rightJS', (0.0, 0.0))

            try:
                vx  =  self.MAX_VEL_X * (ljy)        # up on joystick = +y in Quest
                vy  =  self.MAX_VEL_Y * (-ljx)         # right on joystick = +x in Quest
                wz  =  self.MAX_YAW   * (-rjx)         # right on JS -> negative yaw

                # FIX: smooth velocities to reduce jerk
                self._vx_f = self._smooth(self._vx_f, vx, self.VEL_SMOOTH_ALPHA)
                self._vy_f = self._smooth(self._vy_f, vy, self.VEL_SMOOTH_ALPHA)
                self._wz_f = self._smooth(self._wz_f, wz, self.VEL_SMOOTH_ALPHA)

                self.spot.move_base_with_velocity(vx, vy, wz)

            except Exception as e:
                print(f"[!] Error sending base command: {e}")

            # ---------------- GRIPPER TRIGGER ----------------------
            trigger_val = 1- buttons.get('rightTrig', (0.0,))[0] # use reverse of right trigger
            self.spot.send_gripper(trigger_val)

            # ---------------- ARM   (RIGHT HAND) -------------------
            try:
                lgrip = buttons.get('leftGrip', (0.0,))[0] > 0.5 or buttons.get('LG', False)
                rmat_raw  = poses['r']
                # Meta Quest arm frame has z-down, x-right coordinate system.
                # Reaxis to convert to robot hand frame: x-forward, y-left, z-up
                rmat = map_controller_to_robot(rmat_raw)

                if lgrip and not self.prev_r_grip:
                    # first frame with grip pressed -> anchor
                    self.arm_anchor_ctrl  = rmat.copy()
                    self.arm_anchor_robot = self.spot.current_ee_pose_se3()
                if lgrip:
                    self.stowed = False  # arm is unstowed when moving arm
                    # Cartesian delta = anchor^{-1} * current
                    # Compute controller delta in anchor frame
                    delta = np.linalg.inv(self.arm_anchor_ctrl) @ rmat

                    delta_scaled = delta.copy()
                    delta_scaled[0:3, 3] *= self.ARM_SCALE  # scale translation
                    delta_pose = mat_to_se3(delta_scaled)

                    # Compose with robot-space anchor to get absolute target
                    goal = self.arm_anchor_robot * delta_pose  # composition


                    # ---- force-aware delta limiting (project onto contact normal) ----
                    PARALLEL_MIN_SCALE = 0.0   # residual motion allowed along normal at/above hard (0..1)
                    F_DETECT = 0.8              # N; ignore tiny forces to avoid flicker
                    ALPHA_F = 0.2               # force EWMA smoothing factor

                    # absolute target (BODY) from your anchor + controller
                    desired_xyz = np.array([goal.x, goal.y, goal.z], dtype=float)

                    # operate on DELTA (not absolute goal)
                    pose_now = self.spot.current_ee_pose_se3()
                    x_now = np.array([pose_now.x, pose_now.y, pose_now.z], dtype=float)

                    # per-tick commanded delta in BODY
                    delta_cmd = desired_xyz - x_now

                    if self.force_limit_enabled:
                        delta_out = self._apply_force_limit(delta_cmd, pose_now, soft_limits, hard_limits)
                    else:
                        delta_out = delta_cmd
                    # convert back to absolute target to send
                    blended_xyz = x_now + delta_out
                        
                        # if not hasattr(self, "_dbg_t"): self._dbg_t = 0.0
                        # try:
                        #     if time.time() - getattr(self, "_dbg_t", 0) > 0.25:
                        #         self._dbg_t = time.time()
                        #         print(f"‖F‖={normF:5.2f}  soft_eff={soft_eff:4.2f} hard_eff={hard_eff:4.2f}  "
                        #             f"d_par={d_par_mag:6.3f}  alpha={alpha:4.2f}  "
                        #             f"F_body={F_body.round(2)}  n={n_c.round(3)}")
                        # except Exception:
                        #     pass
                    cmd_pos = np.array(blended_xyz)
                    cmd_quat = np.array([goal.rot.x, goal.rot.y, goal.rot.z, goal.rot.w])
                    print(f"[ARM CMD] pos={np.round(cmd_pos,4).tolist()} quat={np.round(cmd_quat,4).tolist()}")
                    self.spot.send_arm_cartesian_hybrid(
                        pos_xyz=cmd_pos,
                        quat_xyzw=cmd_quat,
                        seconds=0.25,
                        max_lin_vel=0.35,
                        max_ang_vel=1.5,
                        root_frame="body",  # or "vision"
                    )
                    # update previous AFTER sending
                    self.prev_goal_xyz = blended_xyz.copy()
                

                self.prev_r_grip = lgrip
            except Exception as e:
                print(f"[!] Error sending arm command: {e}")

            # ------------- keep teleop RT responsive --------------
            sleep = max(0.0, dt - (time.time() - t_prev))
            time.sleep(sleep)
            t_prev = time.time()

            if self.demo_image_preview:
                self.recorder.poll_preview()


def main():
    robot_ip = os.environ.get("SPOT_ROBOT_IP", "192.168.1.138")
    user     = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
    password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")

    meta_ip = "192.168.1.35" #get_connecteed_device_ip()
    if meta_ip is None:
        print("[!] Could not find connected Meta Quest device IP via ADB.")
        print("    Please ensure ADB is set up and allowed access.")
        return
    
    print(f"Connecting to Spot at {robot_ip} ...")
    print(f"user: {user}, password: {len(password) * '*'}")

    home_pose = [0.55, 0.0, 0.55, 0.0, 0.5, 0, 0.8660254]
    # home_pose = [0.6328, 0.0054, 0.3568, -0.7006, -0.1321, -0.018, 0.701]
    teleop = SpotVRTeleop(robot_ip, user, password, home_pose=home_pose, meta_quest_ip= meta_ip, demo_image_preview=False)
    teleop.run()

if __name__ == "__main__":
    main()
