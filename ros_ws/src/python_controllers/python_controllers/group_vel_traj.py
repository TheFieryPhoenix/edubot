try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
except ImportError:
    rclpy = None
    # dummy base class so file can be imported outside of ROS
    class Node:
        def __init__(self, *args, **kwargs):
            pass

import numpy as np
import sympy as sp

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
except ImportError:
    # define dummy message classes for non-ROS import scenarios
    class JointTrajectory:
        def __init__(self):
            self.header = type('h', (), {'stamp': None})
            self.points = []
    class JointTrajectoryPoint:
        def __init__(self):
            self.velocities = []

try:
    from sensor_msgs.msg import JointState
except ImportError:
    class JointState:
        def __init__(self):
            self.name = []
            self.position = []

# ----------------------------
# Basic rotations (symbolic)
# ----------------------------
def rotx(a):
    return sp.Matrix([[1,0,0],[0,sp.cos(a),-sp.sin(a)],[0,sp.sin(a),sp.cos(a)]])

def roty(a):
    return sp.Matrix([[sp.cos(a),0,sp.sin(a)],[0,1,0],[-sp.sin(a),0,sp.cos(a)]])

def rotz(a):
    return sp.Matrix([[sp.cos(a),-sp.sin(a),0],[sp.sin(a),sp.cos(a),0],[0,0,1]])

def rot_z_4(q):
    return sp.Matrix([
        [sp.cos(q), -sp.sin(q), 0, 0],
        [sp.sin(q),  sp.cos(q), 0, 0],
        [0,          0,         1, 0],
        [0,          0,         0, 1]
    ])

def thin_tform(pos, rpy):
    R = rotz(rpy[2]) * roty(rpy[1]) * rotx(rpy[0])
    T = sp.eye(4)
    T[:3,:3] = R
    T[:3,3] = sp.Matrix(pos)
    return T

# ----------------------------
# Build full symbolic transform T_world_gc(q)
# ----------------------------
def get_symbolic_T_world_gc():
    q1,q2,q3,q4,q5 = sp.symbols('q1 q2 q3 q4 q5', real=True)

    # Fixed transforms from assignment (same as your FK)
    T_world_base   = thin_tform([0,0,0], [0,0, sp.pi])  # assignment uses yaw=pi 
    T_base_sh      = thin_tform([0, -0.0452, 0.0165], [0,0,0]) * rot_z_4(q1)
    T_sh_ua        = thin_tform([0, -0.0306, 0.1025], [0, -1.57079, 0]) * rot_z_4(q2)
    T_ua_la        = thin_tform([0.11257, -0.028, 0], [0,0,0]) * rot_z_4(q3)
    T_la_wr        = thin_tform([0.0052, -0.1349, 0], [0,0, sp.pi/2]) * rot_z_4(q4)
    T_wr_gr        = thin_tform([-0.0601, 0, 0], [0, -sp.pi/2, 0]) * rot_z_4(q5)
    T_gr_gc        = thin_tform([0,0,0.075], [0,0,0])

    T = T_world_base * T_base_sh * T_sh_ua * T_ua_la * T_la_wr * T_wr_gr * T_gr_gc
    return T, (q1,q2,q3,q4,q5)

# ----------------------------
# Jacobian construction
# ----------------------------
def get_symbolic_jacobian():
    T, q = get_symbolic_T_world_gc()
    p = T[:3, 3]  # end-effector position

    # 1) Linear Jacobian: Jv = d p / d q  (Lecture 4: x=f(q) => δx = J δq) 
    Jv = p.jacobian(sp.Matrix(q))

    # 2) Rotational Jacobian:
    z_local = sp.Matrix([0,0,1])

    # compute intermediate transforms
    q1,q2,q3,q4,q5 = q

    T_world_base   = thin_tform([0,0,0], [0,0, sp.pi])
    T_base_sh_o    = thin_tform([0, -0.0452, 0.0165], [0,0,0])
    T_sh_ua_o      = thin_tform([0, -0.0306, 0.1025], [0, -1.57079, 0])
    T_ua_la_o      = thin_tform([0.11257, -0.028, 0], [0,0,0])
    T_la_wr_o      = thin_tform([0.0052, -0.1349, 0], [0,0, sp.pi/2])
    T_wr_gr_o      = thin_tform([-0.0601, 0, 0], [0, -sp.pi/2, 0])

    T0 = T_world_base
    T1 = T0 * (T_base_sh_o * rot_z_4(q1))
    T2 = T1 * (T_sh_ua_o   * rot_z_4(q2))
    T3 = T2 * (T_ua_la_o   * rot_z_4(q3))
    T4 = T3 * (T_la_wr_o   * rot_z_4(q4))
    T5 = T4 * (T_wr_gr_o   * rot_z_4(q5))

    R_world_j1 = (T0 * T_base_sh_o)[:3,:3]
    R_world_j2 = (T1 * T_sh_ua_o)[:3,:3]
    R_world_j3 = (T2 * T_ua_la_o)[:3,:3]
    R_world_j4 = (T3 * T_la_wr_o)[:3,:3]
    R_world_j5 = (T4 * T_wr_gr_o)[:3,:3]

    z1 = R_world_j1 * z_local
    z2 = R_world_j2 * z_local
    z3 = R_world_j3 * z_local
    z4 = R_world_j4 * z_local
    z5 = R_world_j5 * z_local

    Jw = sp.Matrix.hstack(z1,z2,z3,z4,z5)

    J = sp.Matrix.vstack(Jv, Jw)
    return sp.simplify(J), sp.simplify(Jv), sp.simplify(Jw), q

# ----------------------------
# Numeric helpers
# ----------------------------

def jacobian_numeric_func():
    J, Jv, Jw, q = get_symbolic_jacobian()
    return sp.lambdify(q, J, modules="numpy")


def fk_position_numeric_func():
    T_sym, q_syms = get_symbolic_T_world_gc()
    p_sym = T_sym[:3, 3]
    return sp.lambdify(q_syms, p_sym, modules="numpy")


def damped_pinv(A, lam=1e-2):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    return A.T @ np.linalg.inv(A @ A.T + (lam**2) * np.eye(m))


def cond_number(A, eps=1e-12):
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    if S[-1] < eps:
        return np.inf
    return float(S[0] / S[-1])


def clamp_vec(v, v_max):
    return np.clip(v, -v_max, v_max)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_vec_to_pi(v):
    return wrap_to_pi(np.asarray(v, dtype=float))


# Per-joint position limits [min, max] in radians.
# These are used ONLY to gate outgoing velocity commands.
# Measured state is NEVER clamped – clamping raw sim output causes
# fake errors that spin joints in the wrong direction.
JOINT_LIMITS_RAD = np.array([
    [np.deg2rad(-135.0), np.deg2rad(135.0)],  # q1  shoulder rotation
    [np.deg2rad(-120.0), np.deg2rad(120.0)],  # q2  shoulder pitch
    [np.deg2rad(-120.0), np.deg2rad(120.0)],  # q3  elbow
    [np.deg2rad(-100.0), np.deg2rad(100.0)],  # q4  wrist pitch
    [np.deg2rad(-180.0), np.deg2rad(180.0)],  # q5  wrist roll
], dtype=float)


def clamp_q_to_limits(q):
    """Clamp joint positions to physical limits."""
    return np.clip(np.asarray(q, dtype=float),
                   JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])


def gate_vel_at_limits(q, dq, margin=np.deg2rad(1.0)):
    """Zero out velocity components that would drive a joint past its limit."""
    q   = np.asarray(q,   dtype=float)
    dq  = np.asarray(dq,  dtype=float)
    out = dq.copy()
    lo  = JOINT_LIMITS_RAD[:, 0] + margin
    hi  = JOINT_LIMITS_RAD[:, 1] - margin
    out = np.where((q <= lo) & (out < 0.0), 0.0, out)
    out = np.where((q >= hi) & (out > 0.0), 0.0, out)
    return out


class ExampleTraj(Node):

    def __init__(self):
        super().__init__('example_trajectory')

        # joint home position – clamped to physical limits at construction
        self._HOME = clamp_q_to_limits(np.array([
            np.deg2rad(0), np.deg2rad(70),
            np.deg2rad(-40), np.deg2rad(-60),
            np.deg2rad(0)
        ], dtype=float))

        # state for velocity control
        self._q = self._HOME.copy()
        self._have_joint_state = False
        self._init_done = False
        self._init_tol = np.deg2rad(1.0)
        self._init_kp = 1.5
        self._rect_started = False
        self._rect_center = None
        self._rect_half_y = 0.05
        self._rect_half_z = 0.03
        self._rect_period = 10.0
        self._cart_kp = 1.2
        self._joint_name_candidates = {
            'q1': ['q1', 'Shoulder_Rotation'],
            'q2': ['q2', 'Shoulder_Pitch'],
            'q3': ['q3', 'Elbow'],
            'q4': ['q4', 'Wrist_Pitch'],
            'q5': ['q5', 'Wrist_Roll'],
        }
        self._last_time = self.get_clock().now()

        # numeric kinematics functions (computed once)
        self._J_func = jacobian_numeric_func()
        self._fk_func = fk_position_numeric_func()

        self._beginning = self.get_clock().now()
        self._publisher = self.create_publisher(JointTrajectory, 'joint_cmds', 10)
        self._marker_pub = self.create_publisher(Marker, 'ee_trace', 10)
        self._trace_marker = Marker()
        self._trace_marker.header.frame_id = 'world'
        self._trace_marker.ns = 'ee_trace'
        self._trace_marker.id = 0
        self._trace_marker.type = Marker.LINE_STRIP
        self._trace_marker.action = Marker.ADD
        self._trace_marker.scale.x = 0.003
        self._trace_marker.color.r = 0.0
        self._trace_marker.color.g = 0.8
        self._trace_marker.color.b = 1.0
        self._trace_marker.color.a = 1.0
        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            joint_state_qos,
        )
        timer_period = 0.04  # seconds
        self._timer = self.create_timer(timer_period, self.timer_callback)

    def joint_state_callback(self, msg):
        if not msg.name or not msg.position:
            return

        name_to_idx = {name: index for index, name in enumerate(msg.name)}
        q_new = self._q.copy()

        ordered_keys = ['q1', 'q2', 'q3', 'q4', 'q5']
        for joint_index, key in enumerate(ordered_keys):
            candidates = self._joint_name_candidates[key]
            selected_idx = None
            for candidate in candidates:
                if candidate in name_to_idx:
                    selected_idx = name_to_idx[candidate]
                    break
            if selected_idx is None or selected_idx >= len(msg.position):
                return
            # wrap to [-pi, pi] so accumulated velocity-mode positions
            # are normalised; do NOT clamp – clamping causes state errors
            q_new[joint_index] = float(wrap_to_pi(msg.position[selected_idx]))

        self._q = q_new
        self._have_joint_state = True

    def timer_callback(self):
        now = self.get_clock().now()
        msg = JointTrajectory()
        msg.header.stamp = now.to_msg()

        # compute elapsed times
        t_total = (now - self._beginning).nanoseconds * 1e-9
        self._last_time = now

        point = JointTrajectoryPoint()

        # wait for first feedback before moving
        if not self._have_joint_state:
            self.get_logger().warn(
                'waiting for joint_states…', throttle_duration_sec=2.0)
            for _ in range(len(self._HOME)):
                point.velocities.append(0.0)
            point.velocities.append(0.0)
            msg.points = [point]
            self._publisher.publish(msg)
            return

        # --- homing phase: drive joints straight to HOME within limits ---
        if not self._init_done:
            # direct error – no angle wrapping; all limits are < 180° so
            # the shortest path is always the direct one
            q_err = self._HOME - self._q
            if np.max(np.abs(q_err)) < self._init_tol:
                self._init_done = True
                self._beginning = now
                self.get_logger().info(
                    'homing complete; starting Cartesian velocity trajectory')
            else:
                dq_home = clamp_vec(self._init_kp * q_err, 0.8)
                dq_home = gate_vel_at_limits(self._q, dq_home)
                for rate in dq_home:
                    point.velocities.append(float(rate))
                point.velocities.append(0.0)
                msg.points = [point]
                self._publisher.publish(msg)
                return

        # desired end-effector twist (vx,vy,vz, wx,wy,wz)
        # closed-loop rectangle tracking in yz-plane (x fixed).
        p_now = np.array(self._fk_func(*self._q), dtype=float).reshape(3)
        if not self._rect_started:
            # start on bottom-left corner so there is no initial jump
            self._rect_center = np.array([
                p_now[0],
                p_now[1] + self._rect_half_y,
                p_now[2] + self._rect_half_z
            ])
            self._rect_started = True

        seg_t = self._rect_period / 4.0
        edge_speed_y = (2.0 * self._rect_half_y) / seg_t
        edge_speed_z = (2.0 * self._rect_half_z) / seg_t
        t_mod = t_total % self._rect_period

        cx, cy, cz = self._rect_center
        y_min, y_max = cy - self._rect_half_y, cy + self._rect_half_y
        z_min, z_max = cz - self._rect_half_z, cz + self._rect_half_z

        if t_mod < seg_t:
            s = t_mod / seg_t
            py_des = y_min + (y_max - y_min) * s
            pz_des = z_min
            vy_ff = edge_speed_y
            vz_ff = 0.0
        elif t_mod < 2.0 * seg_t:
            s = (t_mod - seg_t) / seg_t
            py_des = y_max
            pz_des = z_min + (z_max - z_min) * s
            vy_ff = 0.0
            vz_ff = edge_speed_z
        elif t_mod < 3.0 * seg_t:
            s = (t_mod - 2.0 * seg_t) / seg_t
            py_des = y_max - (y_max - y_min) * s
            pz_des = z_max
            vy_ff = -edge_speed_y
            vz_ff = 0.0
        else:
            s = (t_mod - 3.0 * seg_t) / seg_t
            py_des = y_min
            pz_des = z_max - (z_max - z_min) * s
            vy_ff = 0.0
            vz_ff = -edge_speed_z

        p_des = np.array([cx, py_des, pz_des])
        p_err = p_des - p_now
        vx_ff = 0.0

        vx = vx_ff + self._cart_kp * p_err[0]
        vy = vy_ff + self._cart_kp * p_err[1]
        vz = vz_ff + self._cart_kp * p_err[2]
        wx = 0.0
        wy = 0.0
        wz = 0.0
        v_des = np.array([vx, vy, vz, wx, wy, wz])

        # evaluate Jacobian at current joint state
        J = np.array(self._J_func(*self._q), dtype=float)
        # optional diagnostics
        cond = cond_number(J)
        if cond > 1e3:
            self.get_logger().warn(f"high Jacobian condition number {cond:.1e}")

        # compute joint velocities via damped pseudo-inverse
        dq = damped_pinv(J) @ v_des
        dq = clamp_vec(dq, 1.0)          # 1 rad/s max
        dq = gate_vel_at_limits(self._q, dq)  # enforce joint limits

        # form trajectory message using computed joint rates
        for rate in dq:
            point.velocities.append(float(rate))
        # gripper velocity is zero in this example
        point.velocities.append(0.0)

        msg.points = [point]
        self._publisher.publish(msg)

        # publish ee trace marker for RViz
        tp = Point()
        tp.x, tp.y, tp.z = float(p_now[0]), float(p_now[1]), float(p_now[2])
        self._trace_marker.points.append(tp)
        if len(self._trace_marker.points) > 2000:
            self._trace_marker.points.pop(0)
        self._trace_marker.header.stamp = now.to_msg()
        self._marker_pub.publish(self._trace_marker)


def main(args=None):
    rclpy.init(args=args)

    example_traj = ExampleTraj()

    try:
        rclpy.spin(example_traj)
    finally:
        # publish zero velocities on exit so Ctrl+C doesn't leave the robot spinning
        stop_msg = JointTrajectory()
        stop_point = JointTrajectoryPoint()
        stop_point.velocities = [0.0] * 6
        stop_msg.points = [stop_point]
        example_traj._publisher.publish(stop_msg)
        example_traj.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()