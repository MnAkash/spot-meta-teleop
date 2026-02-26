#!/usr/bin/env python3
"""
=================
Natural tele-operation of Boston Dynamics Spot and its arm with Spacemopuse and Keyboard.

The script assumes you already have:
    - Spot SDK 5.0.0 installed in the active conda env
    - A working meta quest setup with adb enabled
    - The `reader.py` module in the same directory, which reads the Meta Quest controller

Author: Moniruzzaman Akash, Devin Borchard
"""
import argparse, math, signal, sys, time, os, threading
import numpy as np
from typing import Tuple, Dict

from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.math_helpers  import SE3Pose, Quat
from spot_teleop.spot_controller import SpotRobotController
import pyspacemouse
from pynput import keyboard

from spot_teleop.demo_recorder import DemoRecorder
from spot_teleop.utils.spot_utils import mat_to_se3, map_controller_to_robot
import logging

keys_pressed_once = set()   # Keys triggered this press
keys_currently_held = set() # Keys that are down right now

def on_press(key):
    try:
        k = key.char
        if k in ['a', 'b', 'x', 'y']:
            if k not in keys_currently_held:
                keys_currently_held.add(k)
                keys_pressed_once.add(k)  # mark as first press event
    except AttributeError:
        pass

def on_release(key):
    try:
        k = key.char
        keys_currently_held.discard(k)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def rpy_to_quat(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return Quat(w, x, y, z)


class SpotVRTeleop:
    MAX_VEL_X = 0.6          # [m/s] forward/back
    MAX_VEL_Y = 0.6          # [m/s] left/right
    MAX_YAW   = 0.8          # [rad/s] spin
    ARM_SCALE = 0.1          # [m per m] controller-to-arm translation

    VEL_SMOOTH_ALPHA = 0.35  # simple first-order smoothing for base
    ARM_SMOOTH_ALPHA = 0.8    # simple first-order smoothing for arm orientations
    DEFAULT_POSE = [0.7, 0, 0.4, 0, 0, 0, 1] # x,y,z, qx,qy,qz,qw

    def __init__(self, robot_ip, username, password, home_pose=None, demo_image_preview=True):
        self.spot = SpotRobotController(robot_ip, username, password)
        self.logger = logging.getLogger("vr-teleop")

        # ---- runtime vars ---
        self.arm_anchor_ctrl  = None   # 4×4 SE(3) when grip first pressed
        self.arm_anchor_robot = None   # SE3Pose   ^ … corresponding robot pose
        self.prev_r_grip      = False
        self.base_enabled     = False
        self.first_frame_arm  = True

        if home_pose == None:
            self.home_pose = self.DEFAULT_POSE
        else:
            self.home_pose = home_pose

        # Base smoothing state
        self._vx_f = 0.0
        self._vy_f = 0.0
        self._wz_f = 0.0

        self._vqx_f = 0.0
        self._vqy_f = 0.0
        self._vqz_f = 0.0
        self._vqw_f = 1.0

        self.stowed = False  # arm stowed at start
        self.demo_image_preview = demo_image_preview  # show camera preview in demo mode1

        self.recorder = DemoRecorder(
            robot=self.spot.robot,
            spot_images=self.spot.spot_images,
            state_client=self.spot.state_client,
            image_size = (320, 240),
            out_dir="demos",
            fps=10,
            preview=self.demo_image_preview)
        
        pyspacemouse.open()
        # SpaceMouse background reader to always keep the latest sample
        self._sm_state = None
        self._sm_lock = threading.Lock()
        self._sm_stop = threading.Event()
        self._sm_thread = threading.Thread(target=self._space_mouse_reader, daemon=True)
        self._sm_thread.start()
        self._last_buttons = [0, 0]

    def get_space_mouse(self):
        with self._sm_lock:
            return self._sm_state

    def _space_mouse_reader(self):
        """Continuously read the SpaceMouse and store the latest state."""
        while not self._sm_stop.is_set():
            state = pyspacemouse.read()
            if state is not None:
                with self._sm_lock:
                    self._sm_state = state
            time.sleep(0.001)

    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.recorder.start()
        else:
            self.recorder.stop()
            self.spot.reset_pose(pose=self.home_pose) # x,y,z, qx,qy,qz,qw

    def _smooth(self, prev: float, target: float, alpha: float) -> float:
        """1st order filter."""
        return (1 - alpha) * prev + alpha * target

    def _deadband(self, val: float, eps: float = 0.02) -> float:
        """Zero small inputs to avoid drift."""
        return 0.0 if abs(val) < eps else val
    
    def run(self):
        rate_hz = 30.0
        dt = 1.0 / rate_hz
        t_prev = time.time()
        self._last_sm_t = None
        
        # --- per-axis limits (N): start scaling at soft, stop by hard ---
        soft_limits = np.array([5.0, 5.0, 5.0], dtype=float)   # Fx, Fy, Fz
        hard_limits = np.array([8.0, 8.0, 8.0], dtype=float)
        self.prev_goal_xyz = None

        # force-delta limiting params (same as VR version)
        PARALLEL_MIN_SCALE = 0.0   # residual motion allowed along normal at/above hard (0..1)
        F_DETECT = 0.8             # N; ignore tiny forces to avoid flicker
        ALPHA_F = 0.2              # force EWMA smoothing factor

        toggle_movement = True     # True = base mode, False = arm mode
        t = 0
        while True:
            action = self.get_space_mouse()

            # --- guard against stale HID samples ---
            if action is None:
                axes = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
                buttons = self._last_buttons
            else:
                stale = self._last_sm_t is not None and abs(action.t - self._last_sm_t) < 1e-6
                self._last_sm_t = action.t

                axes = {
                    "x": action.x,
                    "y": action.y,
                    "z": action.z,
                    "roll": action.roll,
                    "pitch": action.pitch,
                    "yaw": action.yaw,
                }
                buttons = list(action.buttons)

                if stale:
                    axes = {k: 0.0 for k in axes}
                    buttons = self._last_buttons
                else:
                    self._last_buttons = buttons.copy()

            # apply deadband to reduce drift on small inputs
            axes = {k: self._deadband(v) for k, v in axes.items()}

            # ---------------- POWER TOGGLE (A/B) ------------------
            if 'a' in keys_pressed_once:
                keys_pressed_once.remove('a')
                self.spot.dock_undock()

            if 'b' in keys_pressed_once:
                keys_pressed_once.remove('b')
                self.toggle_recording()

            # ------------------ STOW/UNSTOW ARM -------------------
            if 'x' in keys_pressed_once:
                keys_pressed_once.remove('x')
                self.spot.stow_arm()
                toggle_movement = True   # back to base teleop

            if 'y' in keys_pressed_once:
                keys_pressed_once.remove('y') 
                self.spot.reset_pose(pose=self.home_pose)
                toggle_movement = False  # enter arm teleop
                self.first_frame_arm = True   # re-anchor arm when entering arm mode

            # ---------------- BASE  (SpaceMouse translation + yaw) -------------------
            if toggle_movement:
                # whenever we go back to base, force re-anchor of arm next time
                self.first_frame_arm = True

                try:
                    vx = self.MAX_VEL_X * (axes["y"])          # forward/back
                    vy = self.MAX_VEL_Y * (-axes["x"])         # left/right
                    wz = self.MAX_YAW   * (-axes["yaw"])       # yaw

                    # smooth velocities to reduce jerk (same as VR)
                    self._vx_f = self._smooth(self._vx_f, vx, self.VEL_SMOOTH_ALPHA)
                    self._vy_f = self._smooth(self._vy_f, vy, self.VEL_SMOOTH_ALPHA)
                    self._wz_f = self._smooth(self._wz_f, wz, self.VEL_SMOOTH_ALPHA)

                    self.spot.move_base_with_velocity(self._vx_f, self._vy_f, self._wz_f)

                except Exception as e:
                    print(f"[!] Error sending base command: {e}")

            # ---------------- ARM  (SpaceMouse -> Cartesian with force limiting) -----
            else:
                # ---------------- ARM  (SpaceMouse -> incremental Cartesian + RPY + force limiting) -----
                try:
                    # Gripper: use button as binary trigger (invert if you like)
                    trigger_val = 1 - buttons[1] if len(buttons) > 1 else 1
                    self.spot.send_gripper(trigger_val)

                    move_commanded = any(abs(axes[k]) > 0.0 for k in ["x", "y", "z", "roll", "pitch", "yaw"])
                    if not move_commanded:
                        # no motion requested; skip arm command to avoid jitter
                        sleep = max(0.0, dt - (time.time() - t_prev))
                        time.sleep(sleep)
                        t_prev = time.time()
                        if self.demo_image_preview:
                            self.recorder.poll_preview()
                        continue

                    # one-time flag when entering arm mode (kept for future use / re-centering)
                    if self.first_frame_arm:
                        self.first_frame_arm = False
                        self.arm_anchor_robot = self.spot.current_ee_pose_se3()
                        self.prev_goal_xyz = None

                    # --- CURRENT EE POSE / FORCE (single state fetch) ---
                    robot_state = self.spot.state_client.get_robot_state()
                    snap = robot_state.kinematic_state.transforms_snapshot
                    pose_now = get_a_tform_b(snap, BODY_FRAME_NAME, "hand")
                    x_now = np.array([pose_now.x, pose_now.y, pose_now.z], dtype=float)

                    # ---------------- TRANSLATION: incremental from SpaceMouse ----------------
                    # SpaceMouse axes are “how much I want to move now”, not absolute pose
                    delta_cmd = np.array(
                        [
                            axes["y"],    # forward/back
                            -axes["x"],   # left/right
                            axes["z"]     # up/down
                        ],
                        dtype=float
                    ) * self.ARM_SCALE   # tune this gain

                    # desired absolute position before force limiting
                    desired_xyz = x_now + delta_cmd

                    # ----------- FORCE-AWARE DELTA LIMITING (same as VR, only on translation) ----------
                    if np.linalg.norm(delta_cmd) < 1e-9:
                        blended_xyz = desired_xyz
                    else:
                        # --- force in BODY frame ---
                        man = robot_state.manipulator_state
                        fH = man.estimated_end_effector_force_in_hand
                        f_hand = np.array([fH.x, fH.y, fH.z], dtype=float)

                        # BODY <- HAND rotation from CURRENT EE pose
                        try:
                            R_BH = pose_now.rot.to_matrix()  # preferred API
                        except AttributeError:
                            # fallback: build from quaternion
                            qx, qy, qz, qw = pose_now.rot.x, pose_now.rot.y, pose_now.rot.z, pose_now.rot.w
                            R_BH = np.array([
                                [1-2*(qy*qy+qz*qz),   2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                                [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz - qx*qw)],
                                [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)],
                            ], dtype=float)

                        F_body = R_BH @ f_hand

                        # smooth force to avoid chatter
                        self._Ff = (1.0 - ALPHA_F) * getattr(self, "_Ff", np.zeros(3)) + ALPHA_F * F_body
                        F_body = self._Ff
                        normF = float(np.linalg.norm(F_body))

                        # default: pass-through
                        delta_out = delta_cmd

                        if normF > F_DETECT:
                            # contact normal to LIMIT ALONG = INTO the surface = -unit(force)
                            n_c = -F_body / (normF + 1e-9)

                            # decompose commanded delta w.r.t. contact normal
                            d_par_mag = float(np.dot(delta_cmd, n_c))  # >0 = moving INTO contact
                            d_par  = d_par_mag * n_c
                            d_ortho = delta_cmd - d_par

                            if d_par_mag > 0.0:
                                # effective thresholds along n_c (tightest axis wins)
                                abs_nc = np.abs(n_c) + 1e-9
                                soft_eff = float(np.min(soft_limits / abs_nc))
                                hard_eff = float(np.min(hard_limits / abs_nc))

                                # scale α(F): 1 → PARALLEL_MIN_SCALE as force rises from soft→hard
                                if normF <= soft_eff:
                                    alpha = 1.0
                                elif normF >= hard_eff:
                                    alpha = PARALLEL_MIN_SCALE
                                else:
                                    k = (normF - soft_eff) / (hard_eff - soft_eff + 1e-9)
                                    alpha = (1.0 - k) * (1.0 - PARALLEL_MIN_SCALE) + PARALLEL_MIN_SCALE

                                delta_out = d_ortho + alpha * d_par
                            else:
                                # moving away or tangential → don’t restrict
                                delta_out = delta_cmd

                        # convert back to absolute target to send
                        blended_xyz = x_now + delta_out

                    # ---------------- ORIENTATION: incremental RPY from SpaceMouse ----------------
                    # separate rotation gain from translation gain
                    ROT_SCALE = 0.2

                    d_roll  =  axes["roll"]  * ROT_SCALE
                    d_pitch =  axes["pitch"] * ROT_SCALE
                    d_yaw   = -axes["yaw"]   * ROT_SCALE  # sign for yaw to match base convention

                    # small rotation increment as quaternion
                    delta_quat = rpy_to_quat(d_roll, d_pitch, d_yaw)  # your helper; returns a Quat

                    # compose with current orientation
                    # depending on your convention, this might be delta * current or current * delta.
                    # For "apply user twist in hand frame", delta_quat * pose_now.rot is typical.
                    goal_rot = delta_quat * pose_now.rot
                   
                    self._vqx_f = self._smooth(self._vqx_f, goal_rot.x, self.ARM_SMOOTH_ALPHA)
                    self._vqy_f = self._smooth(self._vqy_f, goal_rot.y, self.ARM_SMOOTH_ALPHA)
                    self._vqz_f = self._smooth(self._vqz_f, goal_rot.z, self.ARM_SMOOTH_ALPHA)
                    self._vqw_f = self._smooth(self._vqw_f, goal_rot.w, self.ARM_SMOOTH_ALPHA)
                
                    quat_xyzw = np.array(
                        [self._vqx_f, self._vqy_f, self._vqz_f, self._vqw_f],
                        dtype=float,
                    )

                    # ----------- SEND HYBRID CARTESIAN COMMAND ------------------------
                    print(f"[ARM CMD] pos={blended_xyz.round(4).tolist()} quat={np.round(quat_xyzw,4).tolist()} sec={max(0.02, dt):.3f}")
                    self.spot.send_arm_cartesian_hybrid(
                        pos_xyz=blended_xyz,
                        quat_xyzw=quat_xyzw,
                        seconds=max(0.02, dt),
                        max_lin_vel=0.25,
                        max_ang_vel=1.5,
                        root_frame="body",  # or "vision"
                    )

                    self.prev_goal_xyz = blended_xyz.copy()

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

    print(f"Connecting to Spot at {robot_ip} ...")
    print(f"user: {user}, password: {len(password) * '*'}")
    # 0.55, 0.0, 0.55, 0.0, 0.5, 0, 0.8660254 >> looks isometric top to front
    # 0.7, 0, 0.4, 0, 0, 0, 1 >> looks straight forward
    home_pose = [0.6328, 0.0054, 0.3568, -0.7006, -0.1321, -0.018, 0.701]

    teleop = SpotVRTeleop(robot_ip, user, password, home_pose=home_pose, demo_image_preview=False)
    teleop.run()

if __name__ == "__main__":
    main()
