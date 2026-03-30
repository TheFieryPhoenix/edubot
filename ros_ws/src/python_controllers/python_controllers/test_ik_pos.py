import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class VisualizePoses(Node):

    def __init__(self):
        super().__init__('pose_visualizer_node')

        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        
        # Publish at 25Hz (every 0.04 seconds)
        timer_period = 0.04  
        self._timer = self.create_timer(timer_period, self.timer_callback)

        # Track the active pose so we only print to the console once per change
        self._active_pose_index = -1

        # The 5 calculated poses from the IK solver. 
        # The 6th value (0.0) is added to match the array size of your original script (likely for the gripper).
        self.poses = [
            [-0.9121, -1.1054,  0.4748,  2.2006,  0.0000, 0.0], # Pose I   (Reachable)
            [-1.3034,  0.0966,  1.3270, -1.4236,  0.0000, 0.0], # Pose II  (Unreachable / Stretched)
            [-3.1416,  0.4556,  1.3270, -2.5676,  0.0000, 0.0], # Pose III (Unreachable / Stretched)
            [0.0000, -0.511, -1.210,  -1.421,  -3.14, 0.0], # Pose IV  (Reachable)
            [-1.5708,  0.7079,  1.3270, -2.0349, -0.7850, 0.0]  # Pose V   (Unreachable / Stretched)
        ]

        self.pose_labels = ["Pose I (Reachable)", "Pose II (Unreachable limit)", 
                            "Pose III (Unreachable limit)", "Pose IV (Reachable)", 
                            "Pose V (Unreachable limit)"]

    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        # Calculate total elapsed time in seconds
        dt = (now - self._beginning).nanoseconds * (1e-9)
        
        # Change pose every 10 seconds. Use modulo (%) to loop back to the start.
        current_index = int(dt // 10.0) % len(self.poses)
        
        # Print to the terminal when the pose changes so you know what you are looking at in RViz
        if current_index != self._active_pose_index:
            self.get_logger().info(f"Visualizing: {self.pose_labels[current_index]}")
            self._active_pose_index = current_index

        # Build the trajectory point
        point = JointTrajectoryPoint()
        point.positions = self.poses[current_index]
        
        msg.points = [point]

        # Publish the command
        self._publisher.publish(msg)
        

def main(args=None):
    rclpy.init(args=args)

    pose_visualizer = VisualizePoses()

    try:
        rclpy.spin(pose_visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        pose_visualizer.destroy_node()
        # rclpy.shutdown() may have already been called depending on the executor state
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()