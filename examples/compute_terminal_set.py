import numpy as np
import itertools
import cvxpy as cp

from lib.mpc import MPCStateFB
from lib.environments import *

DT_CONTROL = 0.2  # s
LINEARIZE_STATE = [0, 0, 0, 0, 3]
LINEARIZE_INPUT = [0, 0]


# We are looking for the input invariant set (maximal constraint admissible set)
# Use the (LQR) control law for the unconstrained system, since we define the terminal set to be a set
# within the state constraints.
# use algorithm 1 for `mpc_ex4.pdf` (`mpc_ex2.pdf` gives output admissible set)
# x(k + 1) = (A + BK)x(k)

# VERIFY IN THE END THAT: u within constraints
# DO SO BY: Kx = u --> x is constraint so u is also constraint

def compute_terminal_set(A_k: np.ndarray, A_con: np.ndarray, b_con: np.ndarray, goal: np.ndarray = None) -> tuple[
    np.ndarray, int]:
    _, nx = A_k.shape
    s = len(A_con)
    x_goal = np.zeros(nx) if goal is None else goal

    k_star = None
    stop = False
    for k in itertools.count(1):
        print(f"\n==================== k = {k} ==================== \n")
        # For all i = 1, 2, ..., s
        # Get the (initial) x_i* for a constraint f_i that gives the biggest x at x(k+1)
        # s.t. all constraints on x(0:k)
        x_is = []
        for i in range(s):
            x = cp.Variable(nx)
            # x_offset = x - goal

            # x = np.linalg.matrix_power(A_k, k+1) @ x
            objective = cp.Maximize(A_con[i] @ (np.linalg.matrix_power(A_k, k + 1) @ x) - b_con[i])
            A_term_set = np.vstack(
                [A_con @ np.linalg.matrix_power(A_k, t) for t in range(k + 1)])  # range(k+1), bc also k
            b_term_set = np.hstack((b_con,) * (k + 1))
            constraint = [A_term_set @ x - b_term_set <= 0]
            # for t in range(k):  # --> A_k^{k} x statisfy all these constraints = x0; A_k @ x0 = A_k^{k+1} x --> within constraints --> This is the terminal set
            #     x_t = np.linalg.matrix_power(A_k, t) @ x  # x at timestep t
            #     constraint += [A_con @ x_t - b_con <= 0]  # Apply all constraints at once
            prob = cp.Problem(objective, constraint)
            result = prob.solve(verbose=False)
            print(result)
            x_is.append(x.value)

        # For all i = 1, 2, ..., s
        # if constraint i is satisfied for x_i* for all i
        for i in range(s):
            x_i = x_is[i]
            if x_i is None:
                stop = True
                break
            print((A_con[i] @ np.linalg.matrix_power(A_k, k + 1) @ x_i - b_con[i] <= 0))
            if (A_con[i] @ np.linalg.matrix_power(A_k, k + 1) @ x_i - b_con[i] <= 0).astype(bool):
                if i == s - 1:
                    k_star = k
                    print(f"FOUND K_STAR: {k_star}")
                    return k_star
            else:
                break
        # if k % 10 == 0:
        # print(x_is)
        print(len(A_term_set))
        if stop:
            print("UNSTABLE")
            return k_star
        if k > 100:
            print(x_is)
            print("TO MANY ITERATIONS")
            return k_star


if __name__ == "__main__":
    # Numpy print options
    # np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

    # Set up the environment
    environment = RoadMultipleCarsEnv()

    # Set up the MPC controller
    controller = MPCStateFB(dt=DT_CONTROL, N=1, lin_state=LINEARIZE_STATE, lin_input=LINEARIZE_INPUT,
                            terminal_constraint=False,
                            input_constraint=False, state_constraint=True, env=environment)
    print("============ Controller ============\n")
    A, B, R, P = controller.A, controller.B, controller.R, controller.P
    # Unconstrained control law (LQR) : u = K x
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    # x+ = A x + B (K x) = (A + B K) x = A_k x
    A_k = A + B @ K
    # print(f"A:\n{controller.A}\n")
    # print(f"B:\n{controller.B}\n")
    # print(f"K:\n{K}\n")
    # print(f"A_k:\n{A_k}\n")

    # Compute the constraint matrices
    assert controller.N == 1, "Horizon should be 1"
    A_con, b_con = controller.state_constraint()
    A_con = A_con[..., 5:]  # only use the constraints for x(1) (given that N=1)
    print("\n\n============ Constraints ============\n")
    print(f"A {A_con.shape}:\n{A_con}\n")
    print(f"b {b_con.shape}:\n{b_con}")
    controllability_matrix = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(5)])
    print(f"Eigenvalues A: {np.linalg.eig(A)[0]}")
    print(f"Eigenvalues Ak = A + BK: {np.linalg.eig(A_k)[0]}")
    assert np.all(np.linalg.eig(A_k)[0] <= 1), "EIGENVALUES TO BIG"

    compute_terminal_set(A_k, A_con, b_con)
