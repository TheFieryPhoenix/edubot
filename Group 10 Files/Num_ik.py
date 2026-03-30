import math

# --- Robot Hardware Constants (Derived from URDF) ---
L1 = 0.1160        
ALPHA1 = -0.244    
L2 = 0.1350        
ALPHA2 = -1.532    
Z_OFFSET = 0.119   
L_TOOL = 0.0150    # Corrected based on URDF wrist-to-gripper frame transforms

# --- Joint Limits ---
JOINT_LIMITS = [
    [-2.00, 2.00],   # q1: Base Pan
    [-1.57, 1.57],   # q2: Shoulder Pitch
    [-1.58, 1.58],   # q3: Elbow
    [-1.57, 1.57],   # q4: Wrist Pitch
    [-3.14, 3.14]    # q5: Wrist Roll
]

def solve_ik(x, y, z, pitch, roll):
    valid_configs = []

    # 1. Base Pan (q1)
    if x == 0 and y == 0:
        q1 = 0.0 
    else:
        q1 = math.atan2(x, -y)

    # 2. Wrist Center Decoupling
    r_target = math.sqrt(x**2 + y**2)
    r_j4 = r_target - L_TOOL * math.cos(pitch)
    z_j4 = z - L_TOOL * math.sin(pitch)

    # 3. Calculate Elbow Angle (theta3)
    D_sq = r_j4**2 + (z_j4 - Z_OFFSET)**2
    cos_theta3 = (D_sq - L1**2 - L2**2) / (2 * L1 * L2)

    if cos_theta3 < -1.0 or cos_theta3 > 1.0:
        distance = math.sqrt(D_sq)
        max_reach = L1 + L2
        print(f"    -> REJECTED: Physically out of reach. (Distance to wrist: {distance:.3f}m, Max Reach: {max_reach:.3f}m)")
        return valid_configs 

    theta3_down = math.atan2(math.sqrt(1 - cos_theta3**2), cos_theta3)
    theta3_up = math.atan2(-math.sqrt(1 - cos_theta3**2), cos_theta3)

    for state, theta3 in [("Elbow Down", theta3_down), ("Elbow Up", theta3_up)]:
        
        q3 = theta3 - ALPHA2 + ALPHA1

        # 4. Shoulder Angle (q2)
        gamma = math.atan2(z_j4 - Z_OFFSET, r_j4)
        beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
        theta2 = gamma - beta
        
        # Apply URDF offset AND the pi/2 rotation to account for the arm pointing UP at zero
        q2 = theta2 - (math.pi / 2) - ALPHA1 

        # 5. Wrist Pitch (q4)
        # Offset by pi/2 to align with the URDF's absolute world pitch
        q4 = pitch - (q2 + q3) - (math.pi / 2)

        # 6. Wrist Roll (q5)
        q5 = roll

        # --- Filter against Hardware Limits ---
        q = [q1, q2, q3, q4, q5]
        is_valid = True
        for i in range(5):
            if not (JOINT_LIMITS[i][0] <= q[i] <= JOINT_LIMITS[i][1]):
                print(f"    -> REJECTED ({state}): Joint {i+1} violates limit. Calculated: {q[i]:.3f} rad. Limit: {JOINT_LIMITS[i]}")
                is_valid = False
                break
                
        if is_valid:
            valid_configs.append(q)
            print(f"    -> ACCEPTED ({state})")

    return valid_configs

# --- Main Execution ---
if __name__ == "__main__":
    
    # NEW Reachable Targets for the so_arm100
    targets = [
        {"id": "A (Forward)",  "X": 0.0, "Y": 0.15, "Z": 0.15, "Roll": 0.0, "Pitch": 0.0},
        {"id": "B (Low Grab)", "X": 0.1, "Y": 0.10, "Z": 0.05, "Roll": 0.0, "Pitch": -0.5},
        {"id": "C (Overhead)", "X": 0.0, "Y": 0.05, "Z": 0.25, "Roll": 1.5, "Pitch": 1.57}
    ]

    for t in targets:
        print(f"\n--- Solving Target {t['id']} ---")
        solutions = solve_ik(t["X"], t["Y"], t["Z"], t["Pitch"], t["Roll"])
        
        if solutions:
            print(f"  Final Configurations:")
            for idx, sol in enumerate(solutions):
                formatted_sol = [f"{angle:.3f}" for angle in sol]
                print(f"  Config {idx + 1}: [q1: {formatted_sol[0]}, q2: {formatted_sol[1]}, q3: {formatted_sol[2]}, q4: {formatted_sol[3]}, q5: {formatted_sol[4]}]")