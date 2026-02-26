import math
import sympy as sp

# ==========================================
# 1. FORWARD KINEMATICS HELPER FUNCTIONS
# ==========================================
def rotx(a):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(a), -sp.sin(a)],
        [0, sp.sin(a), sp.cos(a)]
    ])

def roty(a):
    return sp.Matrix([
        [sp.cos(a), 0, sp.sin(a)],
        [0, 1, 0],
        [-sp.sin(a), 0, sp.cos(a)]
    ])

def rotz(a):
    return sp.Matrix([
        [sp.cos(a), -sp.sin(a), 0],
        [sp.sin(a), sp.cos(a), 0],
        [0, 0, 1]
    ])

def rot_z(q):
    return sp.Matrix([
        [sp.cos(q), -sp.sin(q), 0, 0],
        [sp.sin(q),  sp.cos(q), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def thin_tform(pos, rpy):
    R = rotz(rpy[2]) * roty(rpy[1]) * rotx(rpy[0])
    T = sp.eye(4)
    T[:3, :3] = R
    T[:3, 3] = sp.Matrix(pos)
    return T

def get_symbolic_fk():
    """Generates the symbolic Forward Kinematics position equations."""
    q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)

    T_world_base = thin_tform([0, 0, 0], [0, 0, math.pi])
    T_base_shldr = thin_tform([0, -0.0452, 0.0165], [0, 0, 0]) * rot_z(q1)
    T_shldr_upper = thin_tform([0, -0.0306, 0.1025], [0, -1.57079, 0]) * rot_z(q2)
    T_upper_lower = thin_tform([0.11257, -0.028, 0], [0, 0, 0]) * rot_z(q3)
    T_lower_wrist = thin_tform([0.0052, -0.1349, 0], [0, 0, sp.pi/2]) * rot_z(q4)
    T_wrist_grip = thin_tform([-0.0601, 0, 0], [0, -sp.pi/2, 0]) * rot_z(q5)
    T_grip_center = thin_tform([0, 0, 0.075], [0, 0, 0])

    T_fk = (T_world_base * T_base_shldr * T_shldr_upper * T_upper_lower * T_lower_wrist * T_wrist_grip * T_grip_center)

    pos = T_fk[:3, 3]
    pos = sp.expand(pos)
    
    # Return both the position matrix and the tuple of symbols used
    return sp.Matrix([sp.simplify(expr.evalf(4, chop=True)) for expr in pos]), (q1, q2, q3, q4, q5)


# ==========================================
# 2. INVERSE KINEMATICS FUNCTION
# ==========================================
def compute_inverse_kinematics(X, Y, Z, theta_pitch=0.0, roll=0.0):
    """
    Calculates the joint angles (q1 to q5) for a 5-DOF robotic arm 
    to reach a target Cartesian coordinate.
    """
    # --- Step 1: Link Lengths ---
    l2 = 0.1160 # Upper arm length
    l3 = 0.1350 # Lower arm (forearm) length
    l4 = 0.1351 # Distance from wrist to gripper center

    # --- Step 2: Base Angle (q1) and Cartesian to Cylindrical Mapping ---
    y1 = 0.0452 # Offset in the y-direction of the shoulder
    y2 = 0.0306 # Offset from the shoulder to the upper arm
    z1 = 0.0165 # Base to shoulder
    z2 = 0.1025 # Shoulder to upper arm
    
    q1 = math.atan2(Y - y1, X) - (math.pi / 2)
    
    r_ee = math.sqrt(X**2 + (Y - y1)**2)
    r_target = r_ee - y2
    z_target = Z - (z1 + z2)

    # --- Step 3: Kinematic Decoupling (Finding the Wrist Center) ---
    r_w = r_target - l4 * math.cos(theta_pitch)
    z_w = z_target - l4 * math.sin(theta_pitch)

    # --- Step 4: Planar 2R Inverse Kinematics ---
    cos3 = (r_w**2 + z_w**2 - l2**2 - l3**2) / (2 * l2 * l3)
    if cos3 > 1.0 or cos3 < -1.0:
        raise ValueError("Target is out of reach for the given arm configuration.")
    
    cos3 = max(min(cos3, 1.0), -1.0) # Safety clamp
    sin3 = math.sqrt(1 - cos3**2)
    
    theta_3 = math.atan2(sin3, cos3)

    # --- Step 5: Calculating the Shoulder Angle (theta_2) ---
    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2 = math.atan2(z_w, r_w) + math.atan2(k2, k1)

    # --- Step 6: Joint Offsets and Final Mapping ---
    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)

    q2 = theta_2 - theta_2_off
    q3 = theta_2_off - theta_3 - theta_3_off
    q4 = theta_pitch - q2 - q3
    q5 = roll

    return [q1, q2, q3, q4, q5]

# ==========================================
# 3. VERIFICATION SCRIPT
# ==========================================
def main():
    # --- Define Targets ---
    target_X = -0.2
    target_Y = 0.338
    target_Z = 0.136769
    target_pitch = 0.0
    target_roll = 0
 
    
    print("--------------------------------------------------")
    print(f" Target Point: X={target_X}, Y={target_Y}, Z={target_Z}, Pitch={target_pitch}, Roll={target_roll}")
    print("--------------------------------------------------")
    
    # --- 1. Compute Inverse Kinematics ---
    q_vals = compute_inverse_kinematics(target_X, target_Y, target_Z, target_pitch, target_roll)
    
    print("\n Computed Joint Angles (radians) via IK:")
    for i, angle in enumerate(q_vals):
        print(f"  q{i+1}: {angle:.4f} rad ({math.degrees(angle):.2f}°)")
        
    # --- 2. Build Symbolic FK and Verify ---
    print("\n  Building Symbolic Forward Kinematics (this may take a second)...")
    pos_expr, q_syms = get_symbolic_fk()
    
    # Create substitution dictionary: {q1: q_vals[0], q2: q_vals[1], ...}
    subs_dict = {q_syms[i]: q_vals[i] for i in range(5)}
    
    # Substitute values into FK equations
    print(" Testing computed angles in Forward Kinematics...")
    fk_result = pos_expr.subs(subs_dict)
    
    # Format and compare results
    fk_X = float(fk_result[0])
    fk_Y = float(fk_result[1])
    fk_Z = float(fk_result[2])
    
    print("\n--------------------------------------------------")
    print(" Forward Kinematics Verification Result:")
    print("--------------------------------------------------")
    print(f"Target X: {target_X:.4f}  |  FK X: {fk_X:.4f}")
    print(f"Target Y: {target_Y:.4f}  |  FK Y: {fk_Y:.4f}")
    print(f"Target Z: {target_Z:.4f}  |  FK Z: {fk_Z:.4f}")
    
    # Error checking
    error = math.sqrt((target_X - fk_X)**2 + (target_Y - fk_Y)**2 + (target_Z - fk_Z)**2)
    print(f"\nTotal Position Error (Euclidean Distance): {error:.6f} m")

if __name__ == "__main__":
    main()