#!/usr/bin/env python3
"""
camera_streamer.py
------------------
Threaded RealSense camera capture for RGB + depth.
"""
from __future__ import annotations
import threading
import time
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except Exception as e:  # ImportError or missing libs
    rs = None
    _RS_IMPORT_ERROR = e
else:
    _RS_IMPORT_ERROR = None


class CameraStreamer:
    def __init__(
        self,
        *,
        width: int = 320,
        height: int = 240,
        fps: int = 30,
        align_depth_to_color: bool = True,
        device_serial: Optional[str] = None,
    ):
        if rs is None:
            raise RuntimeError(f"pyrealsense2 not available: {_RS_IMPORT_ERROR}")

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.device_serial = device_serial
        self._out_width = self.width
        self._out_height = self.height
        self._stream_width = None
        self._stream_height = None

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if self.device_serial:
            self._config.enable_device(self.device_serial)
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self._align = rs.align(rs.stream.color) if self.align_depth_to_color else None
        self._profile = None
        self._depth_scale = None

        self._thread = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest_color: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None

    @property
    def depth_scale(self) -> Optional[float]:
        return self._depth_scale

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop_evt.clear()
        self._last_error = None
        if not self._ensure_device_available():
            print(f"[CameraStreamer] Warning: {self._last_error}")
            return False
        try:
            self._profile = self._start_with_fallback()
        except Exception as e:
            self._last_error = str(e)
            print(f"[CameraStreamer] Warning: {self._last_error}")
            return False
        try:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
        except Exception:
            self._depth_scale = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True

    def _ensure_device_available(self) -> bool:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            self._last_error = "No RealSense camera detected."
            return False
        if self.device_serial:
            found = False
            for dev in devices:
                try:
                    serial = dev.get_info(rs.camera_info.serial_number)
                except Exception:
                    continue
                if serial == self.device_serial:
                    found = True
                    break
            if not found:
                self._last_error = f"RealSense camera with serial '{self.device_serial}' not found."
                return False
        return True

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._pipeline.stop()
        except Exception:
            pass

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            color = None if self._latest_color is None else self._latest_color.copy()
            depth = None if self._latest_depth is None else self._latest_depth.copy()
        return color, depth

    def show_live(self, window_name: str = "RealSense RGB+Depth") -> None:
        color, depth = self.get_latest()
        if color is None or depth is None:
            placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Waiting for camera frames...",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if self._last_error:
                msg = f"Last error: {self._last_error[:80]}"
                cv2.putText(
                    placeholder,
                    msg,
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 128, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, placeholder)
            return

        depth_vis = self._depth_to_color(depth)
        if depth_vis.shape[:2] != color.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([color, depth_vis], axis=1)
        cv2.imshow(window_name, combined)

    # ---------------- worker ---------------- #
    def _worker(self) -> None:
        while not self._stop_evt.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                if self._align is not None:
                    frames = self._align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                if (self._stream_width, self._stream_height) != (self._out_width, self._out_height):
                    color = cv2.resize(color, (self._out_width, self._out_height), interpolation=cv2.INTER_AREA)
                    depth = cv2.resize(depth, (self._out_width, self._out_height), interpolation=cv2.INTER_NEAREST)
                with self._lock:
                    self._latest_color = color
                    self._latest_depth = depth
                    self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                time.sleep(0.01)

    def _depth_to_color(self, depth: np.ndarray) -> np.ndarray:
        depth = depth.astype(np.float32)
        if self._depth_scale is not None:
            depth *= self._depth_scale  # meters
        # Clamp to a reasonable range for visualization.
        depth = np.clip(depth, 0.0, 2.0)
        depth_u8 = (depth / 2.0 * 255.0).astype(np.uint8)
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    def _start_with_fallback(self):
        candidates = [
            (self.width, self.height, self.fps),
            (640, 480, self.fps),
            (848, 480, self.fps),
            (1280, 720, self.fps),
        ]
        last_err = None
        for w, h, fps in candidates:
            config = rs.config()
            if self.device_serial:
                config.enable_device(self.device_serial)
            config.enable_stream(rs.stream.color, int(w), int(h), rs.format.bgr8, int(fps))
            config.enable_stream(rs.stream.depth, int(w), int(h), rs.format.z16, int(fps))
            try:
                profile = self._pipeline.start(config)
                self._config = config
                self._stream_width, self._stream_height = int(w), int(h)
                print(
                    f"[CameraStreamer] Using stream {self._stream_width}x{self._stream_height}@{int(fps)} "
                    f"-> output {self._out_width}x{self._out_height}"
                )
                return profile
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Couldn't resolve requests for supported modes. Last error: {last_err}")


def main() -> None:
    cam = CameraStreamer()
    started = cam.start()
    if not started:
        print("[CameraStreamer] Realsense camera not started.")
        return

    print("Press 'q' to quit.")
    try:
        while True:
            cam.show_live()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
