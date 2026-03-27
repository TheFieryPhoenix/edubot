#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rcl_interfaces.msg import SetParametersResult  # <-- REQUIRED FOR LIVE TUNING

# ==========================================
# 1. RELATIVE INVERSE KINEMATICS FUNCTION
# ==========================================
def get_offset_angles(base_place_angles, z_offset_meters):
    q1_base, q2_base, q3_base, q4_base, q5_base, gripper = base_place_angles
    
    if z_offset_meters == 0.0:
        return base_place_angles 
        
    l2 = 0.1160
    l3 = 0.1350
    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)
    
    theta_2 = q2_base + theta_2_off
    theta_3 = theta_2_off - q3_base - theta_3_off
    pitch = q2_base + q3_base + q4_base
    
    r_w = l2 * math.cos(theta_2) + l3 * math.cos(theta_2 - theta_3)
    z_w = l2 * math.sin(theta_2) + l3 * math.sin(theta_2 - theta_3)
    
    z_w_new = z_w + z_offset_meters
    
    cos3 = (r_w**2 + z_w_new**2 - l2**2 - l3**2) / (2 * l2 * l3)
    if cos3 > 1.0 or cos3 < -1.0:
        raise ValueError(f"Target offset +{z_offset_meters}m is out of reach.")
        
    sin3 = math.sqrt(1 - cos3**2)
    theta_3_new = math.atan2(sin3, cos3)
    
    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2_new = math.atan2(z_w_new, r_w) + math.atan2(k2, k1)
    
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
        # DECLARE LIVE PARAMETERS (Default values)
        # ---------------------------------------------------------
        # Base Rotations (q1)
        self.declare_parameter('q1_pick', -0.1795)
        self.declare_parameter('q1_place', 0.9100)
        
        # Sequence Timings (Seconds)
        self.declare_parameter('time_pick_sweep', 1.5)
        self.declare_parameter('time_plunge', 0.7)
        self.declare_parameter('time_bite', 0.5)
        self.declare_parameter('time_place_sweep', 2.0)
        self.declare_parameter('time_drop', 1.0)
        self.declare_parameter('time_retreat', 0.7)

        # Register the callback to listen for live changes from the GUI
        self.add_on_set_parameters_callback(self.parameters_callback)

        # ---------------------------------------------------------
        # Hardware Configuration Constants
        # ---------------------------------------------------------
        self.GRIPPER_OPEN = 0.6
        self.GRIPPER_CLOSED = 0.3
        self.PLACE_TILT = -0.7480
        
        # State tracking
        self.sequence = []
        self.current_step = 0
        self.start_config = [0.1733, 1.1720, -1.1520, -1.1950, -1.5355, self.GRIPPER_OPEN] # HOME
        self.start_time = self.get_clock().now()

        # Generate the initial sequence before starting the timer
        self.generate_sequence()

        self._timer = self.create_timer(0.02, self.timer_callback)
        self.get_logger().info("Node started. Awaiting sequence execution...")

    def parameters_callback(self, params):
        """This triggers instantly whenever you change a value in the rqt window."""
        for param in params:
            self.get_logger().info(f"Live Update: {param.name} changed to {param.value}")
            
        # Re-generate the sequence list using the newly updated parameters
        self.generate_sequence()
        return SetParametersResult(successful=True)

    def generate_sequence(self):
        """Builds the entire pick and place math array using the current live parameters."""
        
        # Read current live values from ROS 2
        q1_pick = self.get_parameter('q1_pick').value
        q1_place = self.get_parameter('q1_place').value
        
        t_pick_sweep = self.get_parameter('time_pick_sweep').value
        t_plunge = self.get_parameter('time_plunge').value
        t_bite = self.get_parameter('time_bite').value
        t_place_sweep = self.get_parameter('time_place_sweep').value
        t_drop = self.get_parameter('time_drop').value
        t_retreat = self.get_parameter('time_retreat').value

        # Build Waypoints with the live q1_pick
        HOME       = [ 0.1733,  1.1720, -1.1520, -1.1950, -1.5355, self.GRIPPER_OPEN]
        PRE_GRASP  = [q1_pick, -0.0460, -0.4556, -1.1244, -1.5300, self.GRIPPER_OPEN]
        GRASP_0    = [q1_pick, -0.4571, -0.2807, -1.1183, -1.5300, self.GRIPPER_CLOSED]
        POST_GRASP = [q1_pick, -0.0460, -0.4556, -1.1244, -1.5300, self.GRIPPER_CLOSED]

        # Build standard drop point with live q1_place
        PLACE_1 = [q1_place, 0.2715, -1.1137, -1.0324, self.PLACE_TILT, self.GRIPPER_CLOSED]

        new_sequence = []
        
        for i in range(5):
            block_num = i + 1
            z_offset = i * 0.020 
            
            try:
                target_place = get_offset_angles(PLACE_1, z_offset)
                hover_angles = get_offset_angles(target_place, 0.040)
            except ValueError:
                # Fallbacks if IK fails
                target_place = HOME 
                hover_angles = HOME
                
            DYNAMIC_PRE_PLACE  = hover_angles[:5] + [self.GRIPPER_CLOSED]
            DYNAMIC_POST_PLACE = hover_angles[:5] + [self.GRIPPER_OPEN]

            new_sequence.extend([
                # Pick Phase
                (PRE_GRASP,  t_pick_sweep, f"[Block {block_num}] Sweeping to pick zone"),
                (GRASP_0,    t_plunge,     f"[Block {block_num}] Plunge"),
                (POST_GRASP, t_bite,       f"[Block {block_num}] Bite & Lift"),
                
                # Place Phase
                (DYNAMIC_PRE_PLACE,  t_place_sweep, f"[Block {block_num}] Sweeping to Stack"),
                (target_place,       t_drop,        f"[Block {block_num}] Gentle Drop"),
                (DYNAMIC_POST_PLACE, t_retreat,     f"[Block {block_num}] Fast Retreat & Release")
            ])
            
        new_sequence.append((HOME, 2.0, "Time's up! Returning Home"))
        
        # Safely overwrite the active sequence
        self.sequence = new_sequence


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
            self.get_logger().info(f"Starting: {next_label}")
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