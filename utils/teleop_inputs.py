#!/usr/bin/env python3
"""Input helpers for teleop modes: Meta Quest, keyboard, and SpaceMouse."""
from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

from reader import OculusReader

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

try:
    import pyspacemouse
except Exception:
    pyspacemouse = None


class MetaInputHelper:
    def __init__(self, meta_quest_ip=None):
        self.reader = OculusReader(ip_address=meta_quest_ip)

    def get(self):
        return self.reader.get_transformations_and_buttons()


class KeyboardInputHelper:
    ACTION_KEYS = {"a", "b", "x", "y"}
    BASE_KEYS = {"i", "j", "k", "l", "u", "o"}
    SPECIAL_KEYS = {"space"}

    def __init__(self, enable_base_keys: bool = False):
        self.enable_base_keys = bool(enable_base_keys)
        self._once = set()
        self._held = set()
        self._lock = threading.Lock()
        self._listener = None

    def start(self):
        if pynput_keyboard is None:
            raise RuntimeError("pynput not available.")
        if self._listener is not None:
            return
        self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _on_press(self, key):
        k = None
        try:
            k = key.char.lower()
        except AttributeError:
            if pynput_keyboard is not None and key == pynput_keyboard.Key.space:
                k = "space"
        if k is None:
            return
        watch_keys = self.ACTION_KEYS | (self.BASE_KEYS if self.enable_base_keys else set()) | (
            self.SPECIAL_KEYS if self.enable_base_keys else set()
        )
        if k not in watch_keys:
            return
        with self._lock:
            if k not in self._held:
                self._held.add(k)
                if k in self.ACTION_KEYS or k in self.SPECIAL_KEYS:
                    self._once.add(k)

    def _on_release(self, key):
        k = None
        try:
            k = key.char.lower()
        except AttributeError:
            if pynput_keyboard is not None and key == pynput_keyboard.Key.space:
                k = "space"
        if k is None:
            return
        with self._lock:
            self._held.discard(k)

    def consume_once(self, name: str) -> bool:
        k = str(name).lower()
        with self._lock:
            if k in self._once:
                self._once.remove(k)
                return True
        return False

    def consume_action(self, name: str) -> bool:
        return self.consume_once(name)

    def get_base_motion(self) -> Tuple[float, float, float]:
        """
        Returns normalized commanded motion from held keys:
        forward(+)/back(-), right(+)/left(-), yaw_right(+)/yaw_left(-).
        """
        if not self.enable_base_keys:
            return 0.0, 0.0, 0.0
        with self._lock:
            held = set(self._held)
        forward = float(("i" in held) - ("k" in held))
        right = float(("l" in held) - ("j" in held))
        yaw_right = float(("o" in held) - ("u" in held))
        return forward, right, yaw_right


class SpaceMouseInputHelper:
    def __init__(self, deadband: float = 0.02):
        if pyspacemouse is None:
            raise RuntimeError("pyspacemouse not available.")
        self.deadband = float(deadband)
        pyspacemouse.open()
        self._sm_state = None
        self._sm_lock = threading.Lock()
        self._sm_stop = threading.Event()
        self._sm_thread = threading.Thread(target=self._reader, daemon=True)
        self._sm_thread.start()
        self._last_buttons = [0, 0]
        self._last_t = None

    def stop(self):
        self._sm_stop.set()
        if self._sm_thread.is_alive():
            self._sm_thread.join(timeout=1.0)

    def _reader(self):
        while not self._sm_stop.is_set():
            state = pyspacemouse.read()
            if state is not None:
                with self._sm_lock:
                    self._sm_state = state
            time.sleep(0.001)

    def _deadband(self, val: float) -> float:
        return 0.0 if abs(val) < self.deadband else float(val)

    def get_axes_buttons(self) -> tuple[Dict[str, float], list[int]]:
        with self._sm_lock:
            action = self._sm_state

        if action is None:
            axes = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            return axes, list(self._last_buttons)

        stale = self._last_t is not None and abs(action.t - self._last_t) < 1e-6
        self._last_t = action.t
        axes = {
            "x": self._deadband(action.x),
            "y": self._deadband(action.y),
            "z": self._deadband(action.z),
            "roll": self._deadband(action.roll),
            "pitch": self._deadband(action.pitch),
            "yaw": self._deadband(action.yaw),
        }
        buttons = list(action.buttons)
        if stale:
            axes = {k: 0.0 for k in axes}
            buttons = list(self._last_buttons)
        else:
            self._last_buttons = list(buttons)
        return axes, buttons
