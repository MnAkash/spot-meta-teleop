import numpy as np
import cv2
from typing import Dict, Optional, List
from bosdyn.api import geometry_pb2
from bosdyn.client.math_helpers  import SE3Pose, Quat
from bosdyn.api import image_pb2, geometry_pb2

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to rotation matrix (3,3).
    (4,) -> (3,3).
    """
    x, y, z, w = q
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    return np.array([
        [1 - ty * y - tz * z, tx * y - tz * w,     tx * z + ty * w],
        [tx * y + tz * w,     1 - tx * x - tz * z, ty * z - tx * w],
        [tx * z - ty * w,     ty * z + tx * w,     1 - tx * x - ty * y]
    ], dtype=np.float32)

def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (4,) (x,y,z,w)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22

    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    6-D rep -> (3,3).
    6-D representation is a rotation matrix is the first two columns of the rotation matrix.
    """
    a1, a2 = rot6d[:3], rot6d[3:]
    b1 = a1 / np.linalg.norm(a1)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj)
    b3 = np.cross(b1, b2)
    return np.column_stack((b1, b2, b3)).astype(np.float32)

def mat_to_se3(mat: np.ndarray) -> SE3Pose:
    """4x4 numpy SE(3) → bosdyn SE3Pose (SDK 5.x signature)."""
    pos  = mat[:3, 3]
    x, y, z, w = matrix_to_quat(mat[:3, :3])
    quat = Quat(w=w, x=x, y=y, z=z)
    return SE3Pose(pos[0], pos[1], pos[2], quat)   # ← explicit tx, ty, tz , w, x, y, z


def get_trasnformation_mat(x,y,z, tx=0, ty=0, tz=0) -> np.ndarray:
    """
    Get transformation matrix from eular angles and translation in 'xyz'(intrinsic) convention.
    Angles are in degrees.
    """
    cx, cy, cz = np.cos(np.radians([x,y,z]))
    sx, sy, sz = np.sin(np.radians([x,y,z]))

    R = np.array([
        [cy*cz, -cy*sz, sy],
        [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
        [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
    ], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T

def map_controller_to_robot(delta_controller: np.ndarray) -> np.ndarray:
    """
    Convert delta pose from controller frame to robot frame.
    delta_controller: 4x4 numpy array
    return: delta_robot: 4x4 numpy array

    Controller-frame  →  Desired-hand-frame
    new x  (forward)  = -old y
    new y  (left)     = -old x
    new z  (up)       = -old z
    """
    cTr = get_trasnformation_mat(-150, 0, 90)  # controller to robot

    delta_robot = cTr @ delta_controller @ np.linalg.inv(cTr)
    return delta_robot


def proto_to_cv2(img_resp):
    """Convert a Spot ImageResponse with PIXEL_FORMAT_RGB_U8 to an OpenCV BGR image."""
    rows   = img_resp.shot.image.rows
    cols   = img_resp.shot.image.cols
    data   = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
    rgb    = data.reshape((rows, cols, 3))              # still RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)         # OpenCV wants BGR

def image_to_cv(img_resp: image_pb2.ImageResponse) -> np.ndarray:
    """Robust Spot ImageResponse → OpenCV ndarray (BGR or GRAY)."""
    img = img_resp.shot.image
    fmt = img_resp.shot.image.format
    rows, cols = img.rows, img.cols

    # --- 1. JPEG ---------------------------------------------------------
    if fmt == image_pb2.Image.FORMAT_JPEG:
        buf = np.frombuffer(img.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)       # already BGR

    # --- 2. RAW ----------------------------------------------------------
    # Size of the data tells us how many channels we have.
    chan = len(img.data) // (rows * cols)

    if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        # 16-bit depth
        depth = np.frombuffer(img.data, dtype=np.uint16).reshape(rows, cols)
        return depth                                      # caller decides how to visualise

    if chan == 1:
        gray = np.frombuffer(img.data, dtype=np.uint8).reshape(rows, cols)
        # if you want colourised output for cv2.imshow, convert once here
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if chan == 3:
        rgb = np.frombuffer(img.data, dtype=np.uint8).reshape(rows, cols, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    raise ValueError(f"Unhandled channel count ({chan}) or pixel_format {img.pixel_format}")


def frame_pose(snapshot, child: str) -> Optional[geometry_pb2.SE3Pose]:
    """Return Pose of *child* in its declared parent frame; None if missing."""
    edge = snapshot.child_to_parent_edge_map.get(child)
    return edge.parent_tform_child if edge else None

def pose_to_vec(pose: geometry_pb2.SE3Pose) -> np.ndarray:
    """SE3Pose -> 7-vector [tx,ty,tz,qx,qy,qz,qw]"""
    t = pose.position
    q = pose.rotation
    return np.array([t.x, t.y, t.z, q.x, q.y, q.z, q.w], dtype=np.float32)
