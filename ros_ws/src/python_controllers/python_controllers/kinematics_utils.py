# kinematics_utils.py
import math
import numpy as np

# ---------- Homogeneous transforms ----------

def rot_z_np(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0, 0],
                     [sa,  ca, 0, 0],
                     [ 0,   0, 1, 0],
                     [ 0,   0, 0, 1]], dtype=float)

def thin_tform_np(pos, rpy):
    """
    pos: [x,y,z]
    rpy: [roll, pitch, yaw] in radians
    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)  (same as your SymPy FK).
    """
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=float)
    Ry = np.array([[cy, 0, sy],
                   [ 0, 1,  0],
                   [-sy, 0, cy]], dtype=float)
    Rx = np.array([[1,  0,   0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=float)

    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

# ---------- Forward kinematics for gripper center ----------

def forward_kinematics(q):
    """
    q = [q1..q5] joint angles in radians.
    Returns np.array([x,y,z]) of gripper center in world frame.
    Geometry from assignment parent–child table (world->base->...->grippercenter). [file:4]
    """
    q1, q2, q3, q4, q5 = q

    T_world_base  = thin_tform_np([0, 0, 0],           [0, 0, math.pi])
    T_base_shldr  = thin_tform_np([0, -0.0452, 0.0165],[0, 0, 0])         @ rot_z_np(q1)
    T_shldr_upper = thin_tform_np([0, -0.0306, 0.1025],[0, -1.57079, 0])  @ rot_z_np(q2)
    T_upper_lower = thin_tform_np([0.11257, -0.028, 0],[0, 0, 0])         @ rot_z_np(q3)
    T_lower_wrist = thin_tform_np([0.0052, -0.1349, 0],[0, 0, math.pi/2]) @ rot_z_np(q4)
    T_wrist_grip  = thin_tform_np([-0.0601, 0, 0],     [0, -math.pi/2, 0])@ rot_z_np(q5)
    T_grip_center = thin_tform_np([0, 0, 0.075],       [0, 0, 0])

    T_fk = (T_world_base
            @ T_base_shldr
            @ T_shldr_upper
            @ T_upper_lower
            @ T_lower_wrist
            @ T_wrist_grip
            @ T_grip_center)

    return T_fk[:3, 3].copy()

# ---------- Analytic inverse kinematics (position + 1 pitch + roll) ----------

def compute_inverse_kinematics(X, Y, Z, theta_pitch=0.0, roll=0.0):
    """
    Geometric IK consistent with the above FK.
    Returns [q1..q5] for desired gripper center at (X,Y,Z) and tool pitch/roll.

    theta_pitch: desired pitch about tool Y (in world/FK convention).
    roll:        desired roll about tool X.
    """
    # Effective link lengths
    l2, l3, l4 = 0.1160, 0.1350, 0.1351

    # Offsets from assignment link table
    y1, y2 = 0.0452, 0.0306
    z1, z2 = 0.0165, 0.1025

    # ---- Base joint q1 ----
    # NOTE: the -pi/2 is crucial so FK(IK(X,Y,Z)) ≈ (X,Y,Z)
    q1 = math.atan2(Y - y1, X) - math.pi / 2.0

    # End-effector radius around shoulder and height relative to upperarm origin
    r_ee = math.sqrt(X**2 + (Y - y1)**2)
    r_target = r_ee - y2
    z_target = Z - (z1 + z2)

    # Wrist center via kinematic decoupling
    r_w = r_target - l4 * math.cos(theta_pitch)
    z_w = z_target - l4 * math.sin(theta_pitch)

    # 2R planar IK for upper arm + forearm
    num = r_w**2 + z_w**2 - l2**2 - l3**2
    den = 2.0 * l2 * l3
    cos3 = num / den
    if cos3 < -1.0 or cos3 > 1.0:
        raise ValueError("Target out of reach (planar).")

    cos3 = max(min(cos3, 1.0), -1.0)
    sin3 = math.sqrt(1.0 - cos3**2)
    theta_3 = math.atan2(sin3, cos3)

    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2 = math.atan2(z_w, r_w) + math.atan2(k2, k1)

    # Geometric offsets from link geometry
    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)

    q2 = theta_2 - theta_2_off
    q3 = theta_2_off - theta_3 - theta_3_off
    q4 = theta_pitch - q2 - q3
    q5 = roll

    return [q1, q2, q3, q4, q5]
