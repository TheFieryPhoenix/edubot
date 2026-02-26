import math

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
    # Y-direction offsets
    y1 = 0.0452 # Offset in the y-direction of the shoulder
    y2 = 0.0306 # Offset from the shoulder to the upper arm

    # Z-direction offsets
    z1 = 0.0165 # Base to shoulder
    z2 = 0.1025 # Shoulder to upper arm
    
    # Base angle calculation
    q1 = math.atan2(Y - y1, X) - (math.pi / 2)
    
    # End effector radius around the shoulder
    r_ee = math.sqrt(X**2 + (Y - y1)**2)
    r_target = r_ee - y2 # Radius relative to the upper arm
    z_target = Z - (z1 + z2) # Height relative to the upper arm

    # --- Step 3: Kinematic Decoupling (Finding the Wrist Center) ---
    r_w = r_target - l4 * math.cos(theta_pitch)
    z_w = z_target - l4 * math.sin(theta_pitch)

    # --- Step 4: Planar 2R Inverse Kinematics ---
    # Apply Law of Cosines
    cos3 = (r_w**2 + z_w**2 - l2**2 - l3**2) / (2 * l2 * l3)
    
    # Safety Clamp to prevent math domain errors (imaginary numbers)
    cos3 = max(min(cos3, 1.0), -1.0)
    
    # Calculate sine using the Pythagorean identity
    sin3 = math.sqrt(1 - cos3**2)
    
    # Calculate the planar elbow angle
    theta_3 = math.atan2(sin3, cos3)

    # --- Step 5: Calculating the Shoulder Angle (theta_2) ---
    # Forward kinematics components of the 2-link chain
    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    
    # Calculate the planar shoulder angle
    theta_2 = math.atan2(z_w, r_w) + math.atan2(k2, k1)

    # --- Step 6: Joint Offsets and Final Mapping ---
    # Hardware offset for the upper-to-lower arm link
    theta_2_off = math.atan2(0.11257, 0.028)
    
    # Map theoretical angles to actual motor joints
    q2 = theta_2 - theta_2_off
    q3 = theta_2_off - theta_3
    q4 = theta_pitch - q2 - q3
    q5 = roll

    # Return the final joint array
    return [q1, q2, q3, q4, q5]

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Target Point Inputs
    target_X = 0.15
    target_Y = 0.15
    target_Z = 0.35
    
    # Calculate the angles
    q = compute_inverse_kinematics(target_X, target_Y, target_Z)
    
    # Print neatly formatted results
    print("Computed Joint Angles (radians):")
    for i, angle in enumerate(q):
        print(f"q{i+1}: {angle:.4f} radians")