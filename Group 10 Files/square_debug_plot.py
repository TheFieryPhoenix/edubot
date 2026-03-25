# square_debug_plot.py
import math
import numpy as np
import matplotlib.pyplot as plt

from kinematics_utils import compute_inverse_kinematics, forward_kinematics

def generate_square_points(N_per_edge=50):
    """
    Generate desired Cartesian points along a square in X-Z plane at fixed Y.
    Returns desired_xyz: (N,3) array of points in world frame.
    """
    Y_const = 0.10

    p1 = np.array([0.20, Y_const, 0.10])  # bottom-left
    p2 = np.array([0.35, Y_const, 0.10])  # bottom-right
    p3 = np.array([0.35, Y_const, 0.25])  # top-right
    p4 = np.array([0.20, Y_const, 0.25])  # top-left

    edges = [
        (p1, p2),
        (p2, p3),
        (p3, p4),
        (p4, p1),
    ]

    pts = []
    for start, end in edges:
        for i in range(N_per_edge):
            alpha = i / (N_per_edge - 1)
            pts.append((1 - alpha) * start + alpha * end)
    return np.vstack(pts)  # (N,3)

def sample_trajectory(theta_pitch=0.0, roll=0.0, N_per_edge=50):
    desired_xyz = generate_square_points(N_per_edge)
    q_list = []
    fk_xyz = []

    for X, Y, Z in desired_xyz:
        try:
            q = compute_inverse_kinematics(X, Y, Z,
                                           theta_pitch=theta_pitch,
                                           roll=roll)
            q_list.append(q)
            fk_xyz.append(forward_kinematics(q))
        except ValueError as e:
            # unreachable point – skip
            q_list.append([math.nan]*5)
            fk_xyz.append([math.nan, math.nan, math.nan])

    return desired_xyz, np.array(fk_xyz), np.array(q_list)

def main():
    desired_xyz, fk_xyz, q_list = sample_trajectory(
        theta_pitch=0.0,
        roll=0.0,
        N_per_edge=75
    )

    # Filter out NaNs (if any unreachable points)
    mask = ~np.isnan(fk_xyz).any(axis=1)
    desired_xyz = desired_xyz[mask]
    fk_xyz = fk_xyz[mask]

    fig = plt.figure(figsize=(10,4))

    # 3D plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(desired_xyz[:,0], desired_xyz[:,1], desired_xyz[:,2],
             'k--', label='Desired path')
    ax1.plot(fk_xyz[:,0], fk_xyz[:,1], fk_xyz[:,2],
             'r-', label='FK(IK) path')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('3D path in world frame')
    ax1.legend()
    ax1.set_box_aspect([1,1,1])

    # X–Z projection at constant Y
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(desired_xyz[:,0], desired_xyz[:,2], 'k--', label='Desired')
    ax2.plot(fk_xyz[:,0], fk_xyz[:,2], 'r-', label='FK(IK)')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Z [m]')
    ax2.set_title('Projection in X–Z plane (Y≈0.10 m)')
    ax2.legend()
    ax2.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
