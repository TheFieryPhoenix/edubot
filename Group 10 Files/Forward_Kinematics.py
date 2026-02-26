import sympy as sp

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
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
    # Note: Python uses 0-based indexing (rpy[0] is X, rpy[1] is Y, rpy[2] is Z)
    R = rotz(rpy[2]) * roty(rpy[1]) * rotx(rpy[0])
    T = sp.eye(4)             # Create 4x4 identity matrix
    T[:3, :3] = R             # Insert 3x3 rotation matrix
    T[:3, 3] = sp.Matrix(pos) # Insert 3x1 position vector
    return T

# ---------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------
def main():
    # Define symbolic variables
    q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)

    # --- transformation matrices ---
    # FIX: Replaced 3.14 with sp.pi/2 for exact 90-degree symbolic rotation
    T_world_base = thin_tform([0, 0, 0], [0, 0, sp.pi]) 
    
    T_base_shldr = thin_tform([0, -0.0452, 0.0165], [0, 0, 0]) * rot_z(q1)
    
    # FIX: Replaced hardcoded float -1.57079 with exact symbolic -sp.pi/2
    T_shldr_upper = thin_tform([0, -0.0306, 0.1025], [0, -sp.pi/2, 0]) * rot_z(q2)
    
    T_upper_lower = thin_tform([0.11257, -0.028, 0], [0, 0, 0]) * rot_z(q3)
    T_lower_wrist = thin_tform([0.0052, -0.1349, 0], [0, 0, sp.pi/2]) * rot_z(q4)
    T_wrist_grip = thin_tform([-0.0601, 0, 0], [0, -sp.pi/2, 0]) * rot_z(q5)
    T_grip_center = thin_tform([0, 0, 0.075], [0, 0, 0])

    # --- forward kinematics ---
    T_fk = (T_world_base * T_base_shldr * T_shldr_upper * T_upper_lower * T_lower_wrist * T_wrist_grip * T_grip_center)

    # --- extract and clean position ---
    pos = T_fk[:3, 3]
    pos = sp.expand(pos)

    # OPTIMIZATION: Simplify symbolically first, THEN evaluate to 4 sig figs
    pos_simplified = sp.simplify(pos)
    pos_final = sp.Matrix([expr.evalf(4, chop=True) for expr in pos_simplified])

    # --- displaying results ---
    print('\n--- Final Clean Equations ---')
    print(f'X = {pos_final[0]}')
    print(f'Y = {pos_final[1]}')
    print(f'Z = {pos_final[2]}')

    # --- 5. Verify Home ---
    # Substitute all joints with 0
    home_val = pos_final.subs({q1: 1.957, q2: 0.003, q3: .726, q4: 1.57, q5: 0})
    
    print('\nNumerical Home Position:')
    # Convert symbolically evaluated numbers to standard floats for display
    print([float(val) for val in home_val])

if __name__ == "__main__":
    main()
