#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ==========================================
# 1. RELATIVE INVERSE KINEMATICS FUNCTION
# ==========================================
# ==========================================
# 1. RELATIVE INVERSE KINEMATICS FUNCTION
# ==========================================
def get_offset_angles(base_place_angles, z_offset_meters):
    """
    Takes known physical joint angles, finds their theoretical 2D Cartesian plane,
    adds a Z offset, and calculates the new joint angles required to reach that height
    while keeping the horizontal extension and tool pitch locked perfectly in place.
    """
    q1_base, q2_base, q3_base, q4_base, q5_base, gripper = base_place_angles
    
    if z_offset_meters == 0.0:
        return base_place_angles # No change required
        
    # --- Kinematic Link Lengths & Offsets ---
    l2 = 0.1160
    l3 = 0.1350
    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)
    
    # Step 1: Forward Kinematics
    theta_2 = q2_base + theta_2_off
    theta_3 = theta_2_off - q3_base - theta_3_off
    pitch = q2_base + q3_base + q4_base
    
    # [FIXED] Using minus theta_3 to match the robot's physical URDF joint mapping
    r_w = l2 * math.cos(theta_2) + l3 * math.cos(theta_2 - theta_3)
    z_w = l2 * math.sin(theta_2) + l3 * math.sin(theta_2 - theta_3)
    
    # Step 2: Add the requested Z height
    z_w_new = z_w + z_offset_meters
    
    # Step 3: Inverse Kinematics for the new height
    cos3 = (r_w**2 + z_w_new**2 - l2**2 - l3**2) / (2 * l2 * l3)
    if cos3 > 1.0 or cos3 < -1.0:
        raise ValueError(f"Target height offset of +{z_offset_meters}m is out of physical reach.")
        
    sin3 = math.sqrt(1 - cos3**2)
    theta_3_new = math.atan2(sin3, cos3)
    
    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2_new = math.atan2(z_w_new, r_w) + math.atan2(k2, k1)
    
    # Step 4: Map back to actual physical joint angles
    q2_new = theta_2_new - theta_2_off
    q3_new = theta_2_off - theta_3_new - theta_3_off
    q4_new = pitch - q2_new - q3_new
    
    return [q1_base, q2_new, q3_new, q4_new, q5_base, gripper]

# ==========================================
# 2. PICK AND PLACE NODE
# ==========================================
class PickAndPlaceEduBot(Node):

    def __init__(self):
        super().__init__('pick_and_place_edubot')
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)

        # ---------------------------------------------------------
        # Hardware Configuration
        # ---------------------------------------------------------
        GRIPPER_OPEN = 0.6
        GRIPPER_CLOSED = 0.3

        # Hardcoded Home and Pick positions
        HOME       = [ 0.1733,  1.1720, -1.1520, -1.1950, -1.5355, GRIPPER_OPEN]
        PRE_GRASP  = [-0.1795, -0.0460, -0.4556, -1.1244, -1.5300, GRIPPER_OPEN]
        GRASP_0    = [-0.1795, -0.4571, -0.2807, -1.1183, -1.5300, GRIPPER_CLOSED]
        POST_GRASP = [-0.1795, -0.0460, -0.4556, -1.1244, -1.5300, GRIPPER_CLOSED]

        # ---------------------------------------------------------
        # Dynamic Stacking via Relative IK
        # ---------------------------------------------------------
        PLACE_BASE = 0.9100
        PLACE_TILT = -0.7480
        
        # Your confirmed "Perfect" base standard
        PLACE_1 = [PLACE_BASE, 0.2715, -1.1137, -1.0324, PLACE_TILT, GRIPPER_CLOSED]

        stack_targets = []
        
        # Generate target configurations for 5 blocks (+2cm per block)
        for i in range(5):
            z_offset = i * 0.020 # 0.00m, 0.02m, 0.04m, 0.06m, 0.08m
            try:
                new_angles = get_offset_angles(PLACE_1, z_offset)
                stack_targets.append(new_angles)
                
                # Print the calculated angles to the terminal for debugging
                angles_str = ", ".join([f"{q:.4f}" for q in new_angles])
                self.get_logger().info(f"Calculated IK for Block {i+1} (+{z_offset}m): [{angles_str}]")
                
            except ValueError as e:
                self.get_logger().error(str(e))
                stack_targets.append(HOME) # Fallback if out of reach

        # Calculate Universal Clearances (Hovering 15cm above PLACE_1)
        try:
            hover_angles = get_offset_angles(PLACE_1, 0.150)
            UNIVERSAL_PRE_PLACE  = hover_angles[:5] + [GRIPPER_CLOSED]
            UNIVERSAL_POST_PLACE = hover_angles[:5] + [GRIPPER_OPEN]
        except ValueError:
            self.get_logger().warn("Hover height out of reach, using safe approximation.")
            UNIVERSAL_PRE_PLACE  = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_CLOSED]
            UNIVERSAL_POST_PLACE = [PLACE_BASE, 0.0000, -0.8713, -1.3898, PLACE_TILT, GRIPPER_OPEN]

        # # ---------------------------------------------------------
        # # Sequence Generation
        # # ---------------------------------------------------------
        # self.sequence = [(HOME, 3.0, "Moving to Home")]

        # for i, target_place in enumerate(stack_targets):
        #     block_num = i + 1
        #     self.sequence.extend([
        #         # Pick Phase
        #         (PRE_GRASP,  2.5, f"[Block {block_num}] Moving above pick zone"),
        #         (GRASP_0,    2.0, f"[Block {block_num}] Lowering to pick object"),
        #         (POST_GRASP, 1.5, f"[Block {block_num}] Closing Gripper & Lifting"),
                
        #         # Place Phase
        #         (UNIVERSAL_PRE_PLACE, 3.5, f"[Block {block_num}] Hovering above drop zone"),
        #         (target_place,        3.5, f"[Block {block_num}] Lowering to Stack Level {block_num}"),
        #         (UNIVERSAL_POST_PLACE,3.5, f"[Block {block_num}] Opening Gripper & Lifting safely")
        #     ])

        # self.sequence.append((HOME, 3.0, "Sequence Complete! Returning Home"))
        
        # ---------------------------------------------------------
        # OPTIMIZED Sequence Generation (~6.5s per block)
        # ---------------------------------------------------------
        self.sequence = [] # Skip the initial home to save 3 seconds right away!

        for i, target_place in enumerate(stack_targets):
            block_num = i + 1
            
            # 1. Calculate a Dynamic Hover (Only 4cm above THIS specific block's drop point)
            try:
                hover_angles = get_offset_angles(target_place, 0.040)
            except ValueError:
                hover_angles = get_offset_angles(target_place, 0.010) # Fallback to 1cm if max reach hit
                
            DYNAMIC_PRE_PLACE  = hover_angles[:5] + [GRIPPER_CLOSED]
            
            # 2. Overlap the Gripper opening with the retreat
            # By setting the gripper to OPEN here, the servo will actuate 
            # *during* the upward vertical movement.
            DYNAMIC_POST_PLACE = hover_angles[:5] + [GRIPPER_OPEN]

            self.sequence.extend([
                # --- PICK PHASE ---
                # Fast sweep to pick zone
                (PRE_GRASP,  1.5, f"[Block {block_num}] Sweeping to pick zone"),
                # Quick plunge
                (GRASP_0,    0.7, f"[Block {block_num}] Plunge"),
                # Fast bite and immediate lift
                (POST_GRASP, 0.5, f"[Block {block_num}] Bite & Lift"),
                
                # --- PLACE PHASE ---
                # Long lateral sweep to the dynamic hover point
                (DYNAMIC_PRE_PLACE,  2.0, f"[Block {block_num}] Sweeping to Stack"),
                # Gentle vertical drop to place the block (STABILITY FOCUS)
                (target_place,       1.0, f"[Block {block_num}] Gentle Drop"),
                # Fast vertical retreat while springing the gripper open
                (DYNAMIC_POST_PLACE, 0.7, f"[Block {block_num}] Fast Retreat & Release")
            ])
            
        # Optional: Return home only when the timer is likely up
        self.sequence.append((HOME, 2.0, "Time's up! Returning Home"))
    

        # State tracking
        self.current_step = 0
        self.start_config = HOME 
        self.start_time = self.get_clock().now()

        # Timer running at 50 Hz
        self._timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info(f"Starting Sequence: {self.sequence[0][2]}")

    def smooth_step(self, x):
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def timer_callback(self):
        if self.current_step >= len(self.sequence):
            return

        now = self.get_clock().now()
        target_config, duration, label = self.sequence[self.current_step]
        
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if elapsed >= duration:
            self.get_logger().info(f"Completed: {label}")
            self.current_step += 1
            
            if self.current_step >= len(self.sequence):
                self.get_logger().info("=== Pick and Place Complete! ===")
                self._timer.cancel()
                return
            
            self.start_config = target_config
            self.start_time = now
            next_label = self.sequence[self.current_step][2]
            self.get_logger().info(f"Starting Sequence: {next_label}")
            return

        alpha = elapsed / duration
        alpha = self.smooth_step(alpha)

        current_positions = []
        for i in range(6):
            start_val = self.start_config[i]
            target_val = target_config[i]
            current_positions.append(start_val + alpha * (target_val - start_val))

        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in current_positions]
        msg.points = [point]
        
        self._publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceEduBot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Sequence interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()