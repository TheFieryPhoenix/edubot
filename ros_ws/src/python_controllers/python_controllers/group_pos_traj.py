import rclpy
import numpy as np
import math
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ==========================================
# INVERSE KINEMATICS & LIMITS
# ==========================================
JOINT_LIMITS_RAD = {
    'q1': (math.radians(-135), math.radians(135)),
    'q2': (math.radians(-90), math.radians(90)),
    'q3': (math.radians(-90), math.radians(90)),
    'q4': (math.radians(-90), math.radians(90)),
    'q5': (math.radians(-180), math.radians(180))
}

def validate_joint_limits(q_vals):
    joint_names = ['q1', 'q2', 'q3', 'q4', 'q5']
    for i, q in enumerate(q_vals):
        name = joint_names[i]
        min_rad, max_rad = JOINT_LIMITS_RAD[name]
        q_norm = (q + math.pi) % (2 * math.pi) - math.pi
        if not (min_rad <= q_norm <= max_rad):
            raise ValueError(f"Joint {name} out of bounds.")
    return q_vals

def compute_inverse_kinematics(X, Y, Z, theta_pitch=0.0, roll=0.0):
    l2, l3, l4 = 0.1160, 0.1350, 0.1351
    y1, y2 = 0.0452, 0.0306
    z1, z2 = 0.0165, 0.1025
    
    q1 = math.atan2(Y - y1, X) - (math.pi / 2)
    
    r_ee = math.sqrt(X**2 + (Y - y1)**2)
    r_target = r_ee - y2
    z_target = Z - (z1 + z2)

    r_w = r_target - l4 * math.cos(theta_pitch)
    z_w = z_target - l4 * math.sin(theta_pitch)

    cos3 = (r_w**2 + z_w**2 - l2**2 - l3**2) / (2 * l2 * l3)
    if cos3 > 1.0 or cos3 < -1.0:
        raise ValueError(f"Target out of reach. cos3: {cos3}")
    
    cos3 = max(min(cos3, 1.0), -1.0) 
    sin3 = math.sqrt(1 - cos3**2)
    theta_3 = math.atan2(sin3, cos3)

    k1 = l2 + l3 * cos3
    k2 = l3 * sin3
    theta_2 = math.atan2(z_w, r_w) + math.atan2(k2, k1)

    theta_2_off = math.atan2(0.11257, 0.028)
    theta_3_off = math.atan2(0.0052, 0.1349)

    q2 = theta_2 - theta_2_off
    q3 = theta_2_off - theta_3 - theta_3_off
    q4 = theta_pitch - q2 - q3
    q5 = roll

    return validate_joint_limits([q1, q2, q3, q4, q5])


# ==========================================
# ROS 2 NODE
# ==========================================
class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        
        # --- Paper Drawing Configuration ---
        self.cycle_time = 15.0 # Seconds to draw one full triangle
        
        # Z height is 1cm to represent a piece of paper on the desk. 
        # Adjust up slightly (e.g., 0.02) if your pen is thick or you hit the table.
        self.z_height = 0.06   
        
        # Pitch angled at 45 degrees so it doesn't crash the wrist into the table
        self.pitch = -0.785  # -45 degrees in radians, adjust as needed for your pen/tool      
        
        # Defined to be safely reachable while pointing down
        # Creates a small 6cm x 10cm triangle in front of the robot
        self.p1 = np.array([0.18,  0.05]) 
        self.p2 = np.array([0.18, -0.05])
        self.p3 = np.array([0.24,  0.00])
        
        timer_period = 0.04  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)

    def get_triangle_target(self, dt):
        """Calculates the current (X,Y) target on the triangle based on time."""
        t = dt % self.cycle_time  # Loop the time to keep drawing continuously
        segment_time = self.cycle_time / 3.0
        
        if t < segment_time:
            # Interpolate from P1 to P2
            alpha = t / segment_time
            return (1 - alpha) * self.p1 + alpha * self.p2
        elif t < 2 * segment_time:
            # Interpolate from P2 to P3
            alpha = (t - segment_time) / segment_time
            return (1 - alpha) * self.p2 + alpha * self.p3
        else:
            # Interpolate from P3 to P1
            alpha = (t - 2 * segment_time) / segment_time
            return (1 - alpha) * self.p3 + alpha * self.p1

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self._beginning).nanoseconds * (1e-9)
        
        target_xy = self.get_triangle_target(dt)
        
        try:
            q_vals = compute_inverse_kinematics(
                X=target_xy[0], 
                Y=target_xy[1], 
                Z=self.z_height, 
                theta_pitch=self.pitch, 
                roll=0.0
            )
            
            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()
            
            point = JointTrajectoryPoint()
            point.positions = [float(q) for q in q_vals] + [0.0]
            msg.points = [point]

            self._publisher.publish(msg)
            
        except ValueError as e:
            self.get_logger().warn(f"IK Error at X={target_xy[0]:.3f}, Y={target_xy[1]:.3f}: {e}")

def main(args=None):
    rclpy.init(args=args)
    example_traj = ExampleTraj()
    
    try:
        rclpy.spin(example_traj)
    except KeyboardInterrupt:
        pass
    finally:
        example_traj.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()