#!/usr/bin/env python3
"""
demo_recorder.py  –  Continuous Spot logger for diffusion-policy demos
---------------------------------------------------------------------
• Records at `fps` Hz from `start()` until `stop()`
• Dumps one compressed NPZ per session: images, joint states, vision←body pose, … 
• Optionally previews the hand camera while running
---------------------------------------------------------------------
Quick use inside control_spot.py
---------------------------------------------------------------------
from spot_teleop.demo_recorder import DemoRecorder
...
self.recorder = DemoRecorder(
        robot=self.robot,
        spot_images=self.spot_images,
        state_client=self.state_client,
        out_dir="demos",
        fps=5,
        preview=True)
self.recorder.start()

# … tele-op loop …

def _clean_shutdown(self, *a):
    self.recorder.stop()
    ...
"""
from __future__ import annotations
import time, threading, subprocess, atexit
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, List

from bosdyn.api import image_pb2, geometry_pb2
from bosdyn.client import math_helpers
try:
    from spot_teleop.spot_images import CameraSource
    from spot_teleop.utils.spot_utils import pose_to_vec, frame_pose, image_to_cv
    from spot_teleop.camera_streamer import CameraStreamer
except ImportError:
    from .spot_images import CameraSource
    from .utils.spot_utils import pose_to_vec, frame_pose, image_to_cv
    from .camera_streamer import CameraStreamer


def play_sound(path: str, block = False):
    # -q = quiet, removes console output; omit if you want to see mpg123 logs
    player = subprocess.Popen(["mpg123", "-q", path])
    if block:
        player.wait()

# ------------------------------------------------------------------ #
#  Recorder                                                          #
# ------------------------------------------------------------------ #

class DemoRecorder:

    def __init__(
        self,
        robot,
        spot_images,
        state_client,
        external_camera: Optional[CameraStreamer] = None,
        use_depth: bool = False,
        out_dir: str = "demos",
        fps: float = 10.0,
        preview: bool = False,
        image_size: Optional[tuple[int, int]] = None, # Default is 640x480 from spot hand camera
    ):
        self.robot         = robot
        self.spot_images   = spot_images
        self.state_client  = state_client
        self.external_camera = external_camera
        self.use_depth     = bool(use_depth)
        self.fps           = fps
        self.preview       = preview
        self.image_size    = image_size
        self._image_size_checked = False

        self._latest_frame = None
        self._latest_ext_frame = None
        self._frame_lock = threading.Lock()


        self.out_dir       = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # thread plumbing
        self._stop_evt     = threading.Event()
        self._thread       = threading.Thread(target=self._worker, daemon=True)

        # data buffers (grow until stop())
        self._frames:      List[np.ndarray]     = []
        self._depth_frames: List[np.ndarray]    = []
        self._frames_ext: List[np.ndarray]      = []
        self._depth_frames_ext: List[np.ndarray] = []
        self._state_bufs:  Dict[str, List[np.ndarray]] = {}
        self._joint_names: Optional[np.ndarray] = None
        hand_streams = ["visual", "depth_registered"] if self.use_depth else ["visual"]
        self._hand_camera_sources = [CameraSource("hand", hand_streams)]
        self._save_thread: Optional[threading.Thread] = None

        self.is_recording = False
        self._printed_intrinsics = False
        self._external_camera_active = False
        print(
            f"[DemoRecorder] Recording sources: hand_rgb=enabled, "
            f"hand_depth={'enabled' if self.use_depth else 'disabled'}"
        )
        if self.external_camera is not None:
            print("[DemoRecorder] External camera source requested: enabled")
            try:
                started = self.external_camera.start()
                if started is False:
                    print("[DemoRecorder] External camera start returned False (not active).")
                else:
                    self._external_camera_active = True
                    print("[DemoRecorder] External camera source active: RGB+depth")
                    atexit.register(self.external_camera.stop)
            except Exception as e:
                print(f"[DemoRecorder] External camera start error: {e}")
        else:
            print("[DemoRecorder] External camera source requested: disabled")
    
    def _validate_image_size(self, frame_shape: tuple[int, int]) -> bool:
        if self.image_size is None or self._image_size_checked:
            return True
        src_h, src_w = frame_shape
        req_w, req_h = self.image_size
        if req_w <= 0 or req_h <= 0:
            print("[DemoRecorder] image_size must be positive (w, h).")
            self._stop_evt.set()
            return False
        if req_w > src_w or req_h > src_h:
            print(f"[DemoRecorder] image_size {req_w}x{req_h} exceeds source {src_w}x{src_h}.")
            self._stop_evt.set()
            return False
        if req_w * src_h != req_h * src_w:
            print(f"[DemoRecorder] image_size {req_w}x{req_h} does not match source aspect ratio {src_w}x{src_h}.")
            self._stop_evt.set()
            return False
        self._image_size_checked = True
        return True

    def _resize_frame(self, frame: np.ndarray, is_depth: bool) -> np.ndarray:
        if self.image_size is None:
            return frame
        req_w, req_h = self.image_size
        if frame.shape[:2] == (req_h, req_w):
            return frame
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
        return cv2.resize(frame, (req_w, req_h), interpolation=interp)

    def _print_intrinsics_once(self, img_resp: image_pb2.ImageResponse) -> None:
        if self._printed_intrinsics or img_resp is None:
            return
        src = img_resp.source
        rows, cols = img_resp.shot.image.rows, img_resp.shot.image.cols
        model = src.WhichOneof("camera_models")
        print(f"[DemoRecorder] Camera: {src.name} ({cols}x{rows}), model={model}")
        if model == "pinhole":
            intr = src.pinhole.intrinsics
            fx, fy = intr.focal_length.x, intr.focal_length.y
            cx, cy = intr.principal_point.x, intr.principal_point.y
            print(f"[DemoRecorder] Original intrinsics: fx={fx:.3f} fy={fy:.3f} "
                  f"cx={cx:.3f} cy={cy:.3f}")
        elif model == "pinhole_brown_conrady":
            intr = src.pinhole_brown_conrady.intrinsics
            pin = intr.pinhole_intrinsics
            fx, fy = pin.focal_length.x, pin.focal_length.y
            cx, cy = pin.principal_point.x, pin.principal_point.y
            print(f"[DemoRecorder] Original intrinsics: fx={fx:.3f} fy={fy:.3f} "
                  f"cx={cx:.3f} cy={cy:.3f}")
            dist = (intr.k1, intr.k2, intr.p1, intr.p2, intr.k3)
            print(f"[DemoRecorder] Distortion (original): "
                  f"k1={dist[0]:.6f} k2={dist[1]:.6f} p1={dist[2]:.6f} "
                  f"p2={dist[3]:.6f} k3={dist[4]:.6f}")
        elif model == "kannala_brandt":
            intr = src.kannala_brandt.intrinsics
            pin = intr.pinhole_intrinsics
            fx, fy = pin.focal_length.x, pin.focal_length.y
            cx, cy = pin.principal_point.x, pin.principal_point.y
            print(f"[DemoRecorder] Original intrinsics: fx={fx:.3f} fy={fy:.3f} "
                  f"cx={cx:.3f} cy={cy:.3f}")
            dist = (intr.k1, intr.k2, intr.k3, intr.k4)
            print(f"[DemoRecorder] Distortion (original): "
                  f"k1={dist[0]:.6f} k2={dist[1]:.6f} k3={dist[2]:.6f} k4={dist[3]:.6f}")
        else:
            print("[DemoRecorder] Camera intrinsics not available in response.")
            self._printed_intrinsics = True
            return

        if self.image_size is not None:
            req_w, req_h = self.image_size
            sx = req_w / float(cols)
            sy = req_h / float(rows)
            fx_s, fy_s = fx * sx, fy * sy
            cx_s, cy_s = cx * sx, cy * sy
            print(f"[DemoRecorder] Scaled intrinsics ({req_w}x{req_h}): "
                  f"fx={fx_s:.3f} fy={fy_s:.3f} cx={cx_s:.3f} cy={cy_s:.3f}")
            if model == "pinhole_brown_conrady":
                print("[DemoRecorder] Distortion (scaled): unchanged by resize")
            elif model == "kannala_brandt":
                print("[DemoRecorder] Distortion (scaled): unchanged by resize")
        self._printed_intrinsics = True

    # ---------------- public control ------------------------------------ #
    def start(self):
        if self._thread.is_alive():
            print("[DemoRecorder] Already running.")
            try:
                play_sound('media/denied.mp3')
            except Exception as e:
                print(f"[DemoRecorder] Play sound error: {e}")
            return
        # clear old buffers
        self._frames.clear()
        self._depth_frames.clear()
        self._frames_ext.clear()
        self._depth_frames_ext.clear()
        self._state_bufs.clear()
        self._joint_names = None
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        try:
            play_sound('media/start.mp3', block=True)
        except Exception as e:
                print(f"[DemoRecorder] Play sound error: {e}")
        self._thread.start()

        self.is_recording = True
        print("[DemoRecorder] Started recording.")
        

    def stop(self):
        if self._save_thread is not None and self._save_thread.is_alive():
            print("[DemoRecorder] Save already in progress; ignoring request.")
            return

        print("[DemoRecorder] Stopping...")
        self._stop_evt.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            print("[DemoRecorder] WARNING: worker did not exit—forcing shutdown.")
        try:
            play_sound('media/stop.mp3')
        except Exception as e:
                print(f"[DemoRecorder] Play sound error: {e}")

        # Write to disk on a background thread to avoid blocking teleop / robot activity.
        def _save_job():
            self._flush_to_disk()
            self.is_recording = False
            print("[DemoRecorder] Stopped and saved session.")
            try:
                play_sound('media/saved.mp3')
            except Exception as e:
                print(f"[DemoRecorder] Play sound error: {e}")

        self._save_thread = threading.Thread(target=_save_job, daemon=True)
        self._save_thread.start()

    def poll_preview(self):
        """Call this periodically from the main thread to show the latest frame."""
        if not self.preview:
            return

        if not self.is_recording:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        frame = None
        ext_frame = None
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
            if self._latest_ext_frame is not None:
                ext_frame = self._latest_ext_frame.copy()

        if frame is not None:
            view = frame
            if ext_frame is not None:
                # Show Spot and external feed side by side, matching display height.
                h = frame.shape[0]
                if ext_frame.shape[0] != h:
                    new_w = int(ext_frame.shape[1] * (h / float(ext_frame.shape[0])))
                    ext_frame = cv2.resize(ext_frame, (max(new_w, 1), h), interpolation=cv2.INTER_AREA)
                view = np.concatenate([frame, ext_frame], axis=1)
                try:
                    cv2.destroyWindow("Hand RGB")
                except Exception:
                    pass
                cv2.imshow("Spot + External RGB", view)
            else:
                try:
                    cv2.destroyWindow("Spot + External RGB")
                except Exception:
                    pass
                cv2.imshow("Hand RGB", view)
            cv2.waitKey(1)  # Allows OpenCV to process UI events

    # ---------------- background thread --------------------------------- #
    def _worker(self):
        period = 1.0 / self.fps
        next_t = time.time()
        while not self._stop_evt.is_set():
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += period
            try:
                # Single RPC for hand color + registered depth to reduce overhead.
                img_entries = self.spot_images.get_images_by_cameras(self._hand_camera_sources)
                color_resp, depth_resp = None, None
                if img_entries:
                    for entry in img_entries:
                        if entry.image_type == "visual":
                            color_resp = entry.image_response
                        elif entry.image_type == "depth_registered":
                            depth_resp = entry.image_response

                if not self._printed_intrinsics:
                    self._print_intrinsics_once(color_resp or depth_resp)

                frame = image_to_cv(color_resp) if color_resp else None
                depth_frame = image_to_cv(depth_resp) if depth_resp else None

                if frame is None:
                    raise RuntimeError("Failed to retrieve hand color image.")

                if self.image_size is not None:
                    if not self._validate_image_size(frame.shape[:2]):
                        return
                    frame = self._resize_frame(frame, is_depth=False)
                    if depth_frame is not None:
                        depth_frame = self._resize_frame(depth_frame, is_depth=True)

                if self.use_depth and depth_frame is None:
                    # Keep lengths aligned; fall back to zeros if depth missing.
                    depth_frame = np.zeros(frame.shape[:2], dtype=np.uint16)

                state = self.state_client.get_robot_state()
                self._frames.append(frame)
                if self.use_depth:
                    self._depth_frames.append(depth_frame)

                ext_rgb = None
                if self._external_camera_active:
                    ext_rgb, ext_depth = self.external_camera.get_latest()
                    if ext_rgb is not None:
                        self._frames_ext.append(ext_rgb)
                        if self.use_depth:
                            if ext_depth is not None:
                                self._depth_frames_ext.append(ext_depth)
                            else:
                                self._depth_frames_ext.append(np.zeros(ext_rgb.shape[:2], dtype=np.uint16))
                vecs  = self._extract_state(state)
                for k, v in vecs.items():
                    self._state_bufs.setdefault(k, []).append(v)
                # save for preview
                if self.preview:
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._latest_ext_frame = None if ext_rgb is None else ext_rgb.copy()
            except Exception as e:
                print(f"[DemoRecorder] {e}")

    # ---------------- state extraction ---------------------------------- #
    def _extract_state(self, state) -> Dict[str, np.ndarray]:
        kin       = state.kinematic_state
        snapshot  = kin.transforms_snapshot

        # 1. --- Arm joint angles + velocities --------------------------------------------------
        q, dq, names = [], [], []
        for j in kin.joint_states:
            if j.name.startswith("arm0."):
                names.append(j.name)
                q.append(j.position.value)
                dq.append(j.velocity.value)

        if self._joint_names is None:
            self._joint_names = np.array(names)

        # 2. Hand pose in body frame
        hand_pose = frame_pose(snapshot, "hand")
        hand_vec = pose_to_vec(hand_pose) if hand_pose else np.zeros(7, np.float32)

        # --- vision←body pose -------------------------------------------
        vision_in_body = frame_pose(snapshot, "vision")     # vision expressed in body frame
        if vision_in_body:
            vis_vec = pose_to_vec(vision_in_body)
        else:
            vis_vec = np.zeros(7, np.float32)

        # --- body velocity (vision) -------------------------------------
        vlin = kin.velocity_of_body_in_vision.linear
        vang = kin.velocity_of_body_in_vision.angular
        body_vel = np.array([vlin.x, vlin.y, vlin.z,
                             vang.x, vang.y, vang.z], dtype=np.float32)

        # --- gripper & wrench -------------------------------------------
        man      = state.manipulator_state
        gripper  = np.array([man.gripper_open_percentage/100.0], dtype=np.float32)
        wrench   = np.array([
              man.estimated_end_effector_force_in_hand.x,
              man.estimated_end_effector_force_in_hand.y,
              man.estimated_end_effector_force_in_hand.z], dtype=np.float32)

        # --- timestamp ----------------------------------------
        ts    = kin.acquisition_timestamp.seconds + \
                kin.acquisition_timestamp.nanos * 1e-9

        return dict(
            arm_q = np.array(q,  dtype=np.float32),
            arm_dq= np.array(dq, dtype=np.float32),
            ee_pose = hand_vec,
            ee_force = wrench,
            gripper  = gripper,
            vision_in_body = vis_vec,
            body_vel = body_vel,
            t        = np.array([ts],   dtype=np.float64),
        )

    # ---------------- disk writer -------------------------------------- #
    def _flush_to_disk(self):
        if not self._frames:
            print("[DemoRecorder] Nothing captured - no file written.")
            return

        # Defensive: ensure depth buffer length matches frames.
        if self.use_depth and len(self._depth_frames) < len(self._frames):
            missing = len(self._frames) - len(self._depth_frames)
            filler_shape = self._frames[0].shape[:2]
            filler = [np.zeros(filler_shape, dtype=np.uint16) for _ in range(missing)]
            self._depth_frames.extend(filler)
        save_external = self._external_camera_active and len(self._frames_ext) > 0
        if save_external and len(self._frames_ext) < len(self._frames):
            missing = len(self._frames) - len(self._frames_ext)
            h, w = self._frames_ext[0].shape[:2]
            filler = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(missing)]
            self._frames_ext.extend(filler)
        if self.use_depth and save_external and len(self._depth_frames_ext) < len(self._frames):
            missing = len(self._frames) - len(self._depth_frames_ext)
            if self._depth_frames_ext:
                h, w = self._depth_frames_ext[0].shape[:2]
            else:
                h, w = self._frames_ext[0].shape[:2]
            filler = [np.zeros((h, w), dtype=np.uint16) for _ in range(missing)]
            self._depth_frames_ext.extend(filler)

        # build session dict
        session = {
            "images_0" : np.array(self._frames, dtype=object),   # ragged
            "arm_joint_names" : self._joint_names,
        }
        if self.use_depth:
            session["images_0_depth"] = np.array(self._depth_frames, dtype=object)
        if save_external:
            session["images_1"] = np.array(self._frames_ext, dtype=object)
            if self.use_depth:
                session["images_1_depth"] = np.array(self._depth_frames_ext, dtype=object)
        for k, lst in self._state_bufs.items():
            session[k] = np.stack(lst)

        # choose next chronological file name
        existing = list(self.out_dir.glob("*.npz"))
        nums     = [int(p.stem) for p in existing if p.stem.isdigit()]
        idx      = max(nums) + 1 if nums else 0
        fname    = self.out_dir / f"{idx}.npz"

        np.savez_compressed(fname, **session)
        print(f"[DemoRecorder] Wrote {fname}")
