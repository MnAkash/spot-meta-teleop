#!/usr/bin/env python3

'''
Estimate hand-eye calibration from an H5 demo using ArUco marker poses.

Usage:
    python handeye_from_h5.py path/to/demo.h5
'''
import argparse
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from spot_teleop.utils.spot_utils import quat_to_matrix  # noqa: E402

FOVX_DEG = 60.18
FOVY_DEG = 46.99

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_demo(h5_path: Path, demo_id: int):
    with h5py.File(h5_path, "r") as hf:
        demo_key = f"demo_{demo_id}"
        data_grp = hf.get("data")
        if data_grp is None or demo_key not in data_grp:
            raise ValueError(f"Missing {demo_key} in {h5_path}")
        obs = data_grp[demo_key].get("obs")
        if obs is None:
            raise ValueError(f"Missing obs group in {demo_key}")
        if "images_0" not in obs:
            raise ValueError(f"Missing images_0 in {demo_key}/obs")
        if "eef_pos" not in obs or "eef_quat" not in obs:
            raise ValueError(f"Missing eef_pos/eef_quat in {demo_key}/obs")
        images = obs["images_0"][:]
        eef_pos = obs["eef_pos"][:]
        eef_quat = obs["eef_quat"][:]
        return images, eef_pos, eef_quat


def build_detector(dict_name: str):
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"Unsupported aruco dict '{dict_name}'")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return lambda gray: detector.detectMarkers(gray)
    return lambda gray: cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate hand-eye calibration from an H5 demo using ArUco marker poses."
    )
    parser.add_argument("h5_path", type=Path, help="Path to demo .h5 file")
    parser.add_argument("--demo-id", type=int, default=0, help="demo id (default: 0)")
    parser.add_argument("--marker-id", type=int, default=7, help="ArUco marker id to track")
    parser.add_argument("--marker-size", type=float, default=0.186,
                        help="Marker size in meters (e.g. 0.10 for 10cm)")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_1000",
                        help="ArUco dictionary name (e.g. DICT_4X4_1000)")
    parser.add_argument("--method", type=str, default="tsai",
                        choices=sorted(HAND_EYE_METHODS.keys()),
                        help="Hand-eye calibration method")
    args = parser.parse_args()

    images, eef_pos, eef_quat = load_demo(args.h5_path, args.demo_id)

    h, w = images[0].shape[:2]
    fovx = np.deg2rad(FOVX_DEG)
    fovy = np.deg2rad(FOVY_DEG)
    fx = (w / 2.0) / np.tan(fovx / 2.0)
    fy = (h / 2.0) / np.tan(fovy / 2.0)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    print(f"Using image size {w}x{h} with FOVx={FOVX_DEG} FOVy={FOVY_DEG}")
    print(f"Computed intrinsics: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)

    detect = build_detector(args.aruco_dict)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    used = 0
    for i in range(images.shape[0]):
        frame = images[i]
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, _ = detect(gray)
        if ids is None or len(ids) == 0:
            continue

        ids = ids.flatten()
        match_idx = None
        for idx, marker_id in enumerate(ids):
            if int(marker_id) == args.marker_id:
                match_idx = idx
                break
        if match_idx is None:
            continue

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[match_idx]], args.marker_size, K, dist
        )
        rvec, tvec = rvecs[0], tvecs[0]
        R_tc, _ = cv2.Rodrigues(rvec)
        t_tc = tvec.reshape(3, 1)

        t_bg = eef_pos[i].astype(np.float64)
        q = eef_quat[i].astype(np.float64)
        R_bg = quat_to_matrix(q)

        R_gripper2base.append(R_bg)
        t_gripper2base.append(t_bg)
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)
        used += 1

    print(f"Frames with marker {args.marker_id}: {used}")
    if used < 2:
        print("[ERROR] Need at least 2 frames with the marker to solve hand-eye.")
        return 1

    method = HAND_EYE_METHODS[args.method]
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=method
    )

    T_cam2gripper = np.eye(4, dtype=np.float64)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.reshape(3)

    T_gripper2cam = np.eye(4, dtype=np.float64)
    T_gripper2cam[:3, :3] = R_cam2gripper.T
    T_gripper2cam[:3, 3] = (-R_cam2gripper.T @ t_cam2gripper).reshape(3)

    def _format_np_array(mat: np.ndarray) -> str:
        rows = []
        for row in mat:
            row_str = ", ".join(f"{v:.8f}" for v in row)
            rows.append(f"    [{row_str}]")
        return "np.array([\n" + ",\n".join(rows) + "\n])"

    print("\Gripper -> Camera (gTc):")
    print(_format_np_array(T_cam2gripper))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
