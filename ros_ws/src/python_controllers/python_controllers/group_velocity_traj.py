# ==========================================================
# Task 3.3: Velocity control in simulation (Jacobian-based)
# ==========================================================
import numpy as np
import time
import sympy as sp

def fk_position_numeric_func():
    """
    Returns p(q) = [x,y,z] using the SAME symbolic transform as your Jacobian.
    Good for logging/verification in simulation.
    """
    T_sym, q_syms = get_symbolic_T_world_gc()
    p_sym = T_sym[:3, 3]
    return sp.lambdify(q_syms, p_sym, modules="numpy")


def damped_pinv(A, lam=1e-2):
    """
    Damped least-squares pseudo-inverse:
      A^+ = A^T (A A^T + lam^2 I)^(-1)

    This is more stable near singularities than a plain inverse/pinv,
    consistent with the lecture discussion that singularities lead to blow-ups. [2](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)[1](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    return A.T @ np.linalg.inv(A @ A.T + (lam**2) * np.eye(m))


def cond_number(A, eps=1e-12):
    """Condition number via SVD; large => near singular. [2](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)[1](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)"""
    U, S, Vt = np.linalg.svd(np.asarray(A, dtype=float))
    if S[-1] < eps:
        return np.inf
    return float(S[0] / S[-1])


def clamp_vec(v, v_max):
    """Elementwise clamp to [-v_max, v_max]."""
    return np.clip(v, -v_max, v_max)


# ----------------------------------------------------------
# SIM INTERFACE (replace these two methods with your simulator)
# ----------------------------------------------------------
class SimInterface:
    """
    Replace get_joint_positions() and send_joint_velocities(qdot)
    with the functions/API of your simulated robot.
    """
    def get_joint_positions(self):
        """
        Return current joint angles [q1..q5] in radians from the simulator.
        """
        raise NotImplementedError

    def send_joint_velocities(self, qdot):
        """
        Send joint velocity command [q1dot..q5dot] in rad/s to the simulator.
        """
        raise NotImplementedError


class DummySimInterface(SimInterface):
    """
    Offline dummy simulator so you can test the control loop.
    It integrates qdot internally (ONLY for debugging without the real simulator).
    """
    def __init__(self, q0):
        self.q = np.array(q0, dtype=float)

    def get_joint_positions(self):
        return self.q.copy()

    def send_joint_velocities(self, qdot):
        # dummy: integrate with fixed dt later in loop (we do it outside)
        self.last_qdot = np.array(qdot, dtype=float)


def run_task33_on_sim(
    sim: SimInterface,
    v_des,
    dt=0.02,
    T_total=5.0,
    qdot_max=1.0,
    cond_max=200.0,
    lam=1e-2,
    use_dummy_integration=False
):
    """
    Task 3.3 control loop using Jacobian:
      1) read q from sim
      2) compute Jv(q)
      3) compute qdot = Jv^+ * v_des
      4) clamp + singularity check
      5) send qdot to sim

    This follows the Lecture 4 mapping xdot = J(q) qdot and pseudo-inverse usage. [2](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)
    Avoids singularities per assignment hint + lectures. [1](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)[1](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)
    """
    # Build numeric Jacobian and FK position functions ONCE
    Jf = jacobian_numeric_func()          # 6x5
    pf = fk_position_numeric_func()       # 3x1

    v_des = np.array(v_des, dtype=float).reshape(3,)
    steps = int(T_total / dt)

    q_hist, p_hist, qdot_hist, cond_hist = [], [], [], []

    t_start = time.perf_counter()
    for k in range(steps):
        t_k = time.perf_counter()

        # 1) Read current joints from simulator
        q = np.array(sim.get_joint_positions(), dtype=float).reshape(5,)

        # 2) Compute Jacobian and take linear part
        J = np.array(Jf(*q), dtype=float)     # 6x5
        Jv = J[0:3, :]                        # 3x5

        # 3) Check conditioning (singularity avoidance)
        c = cond_number(Jv)
        if c > cond_max:
            print(f"[STOP] Near singularity: cond(Jv)={c:.1f} > {cond_max}")
            break

        # 4) Compute joint velocities from desired EE linear velocity
        #    qdot = Jv^+ * v_des   (pseudo-inverse idea) [2](https://tuenl-my.sharepoint.com/personal/m_mouwakdie_student_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/Lecture%206%20-%20Trajectory%20Planning.pdf)
        Jv_pinv = damped_pinv(Jv, lam=lam)
        qdot = Jv_pinv @ v_des

        # 5) Safety clamp
        qdot = clamp_vec(qdot, qdot_max)

        # 6) Send to simulator
        sim.send_joint_velocities(qdot)

        # Optional: if you're using DummySimInterface, integrate it here
        if use_dummy_integration and isinstance(sim, DummySimInterface):
            sim.q = sim.q + qdot * dt

        # 7) Log (position from FK is useful for plots/report)
        p = np.array(pf(*q), dtype=float).reshape(3,)
        q_hist.append(q.copy())
        p_hist.append(p.copy())
        qdot_hist.append(qdot.copy())
        cond_hist.append(c)

        # Print every ~0.5s
        if (k % max(1, int(0.5 / dt))) == 0:
            print(f"t={k*dt:5.2f}s  p={p}  v_des={v_des}  cond(Jv)={c:.1f}")

        # 8) Maintain loop timing (best-effort)
        elapsed = time.perf_counter() - t_k
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    return (np.array(q_hist), np.array(p_hist), np.array(qdot_hist), np.array(cond_hist))


# ==========================================================
# MAIN: Example run (replace DummySimInterface with your sim)
# ==========================================================
if __name__ == "__main__":
    # Starting configuration (rad)
    q0 = [-1.3798, -0.1510, -0.5309, -0.8882, 0.0]

    # Constant EE linear velocity (m/s): +x at 2 cm/s
    v_des = [0.02, 0.0, 0.0]

    # --- Use dummy sim for testing ---
    sim = DummySimInterface(q0)

    q_hist, p_hist, qdot_hist, cond_hist = run_task33_on_sim(
        sim=sim,
        v_des=v_des,
        dt=0.02,
        T_total=5.0,
        qdot_max=1.0,
        cond_max=200.0,
        lam=1e-2,
        use_dummy_integration=True  # set False when using your real simulator
    )

    if len(p_hist) > 1:
        print("\nApprox EE displacement:", p_hist[-1] - p_hist[0])
        print("Final q:", q_hist[-1])