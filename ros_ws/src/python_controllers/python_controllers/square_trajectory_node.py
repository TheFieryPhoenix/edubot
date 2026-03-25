# square_trajectory_node.py
import rclpy
import numpy as np
import math
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from python_controllers.kinematics_utils import (
    compute_inverse_kinematics,
    forward_kinematics,
)


class SquareTrajNode(Node):

    def __init__(self):
        super().__init__('square_trajectory')

        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(
            JointTrajectory, 'joint_cmds', 10
        )

        # Marker for end-effector trace in RViz
        self._marker_pub = self.create_publisher(Marker, 'fk_ee', 10)
        self._marker = Marker()
        self._marker.header.frame_id = 'world'
        self._marker.ns = 'ee_trace'
        self._marker.id = 0
        self._marker.type = Marker.LINE_STRIP
        self._marker.action = Marker.ADD
        self._marker.scale.x = 0.005
        self._marker.color.r = 0.0
        self._marker.color.g = 1.0
        self._marker.color.b = 0.0
        self._marker.color.a = 1.0
        self._marker.points = []

        # Total time to complete one loop
        self.cycle_time = 20.0  # seconds

        # Square corners in X–Z at constant Y
        self.Y_const = 0.10
        self.p1 = np.array([0.20, self.Y_const, 0.10])  # bottom-left
        self.p2 = np.array([0.35, self.Y_const, 0.10])  # bottom-right
        self.p3 = np.array([0.35, self.Y_const, 0.25])  # top-right
        self.p4 = np.array([0.20, self.Y_const, 0.25])  # top-left

        # Tool orientation choice (same as in plotting script)
        self.theta_pitch = 0.0
        self.roll = 0.0

        timer_period = 0.04  # 25 Hz
        self._timer = self.create_timer(timer_period, self.timer_callback)

    def get_square_target(self, dt):
        t = dt % self.cycle_time
        segment_time = self.cycle_time / 4.0

        if t < segment_time:
            alpha = t / segment_time
            return (1 - alpha) * self.p1 + alpha * self.p2
        elif t < 2 * segment_time:
            alpha = (t - segment_time) / segment_time
            return (1 - alpha) * self.p2 + alpha * self.p3
        elif t < 3 * segment_time:
            alpha = (t - 2 * segment_time) / segment_time
            return (1 - alpha) * self.p3 + alpha * self.p4
        else:
            alpha = (t - 3 * segment_time) / segment_time
            return (1 - alpha) * self.p4 + alpha * self.p1

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self._beginning).nanoseconds * 1e-9

        target_xyz = self.get_square_target(dt)
        X, Y, Z = target_xyz

        try:
            q_vals = compute_inverse_kinematics(
                X=X, Y=Y, Z=Z,
                theta_pitch=self.theta_pitch,
                roll=self.roll
            )

            msg = JointTrajectory()
            msg.header.stamp = now.to_msg()

            # Fill positions; controller in your setup infers joint order from URDF
            point = JointTrajectoryPoint()
            point.positions = [float(q) for q in q_vals] + [0.0]  # gripper at 0.0
            msg.points = [point]

            self._publisher.publish(msg)

            # Publish marker trace using FK
            ee_pos = forward_kinematics(q_vals)
            p = Point()
            p.x, p.y, p.z = float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])
            self._marker.points.append(p)
            if len(self._marker.points) > 1000:
                self._marker.points.pop(0)
            self._marker.header.stamp = now.to_msg()
            self._marker_pub.publish(self._marker)

        except ValueError as e:
            self.get_logger().warn(
                f"IK error at X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}: {e}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
