import numpy as np
import itertools
import cvxpy as cp
import matplotlib.pyplot as plt
import polytope as pc

from lib.mpc import MPCStateFB
from lib.environments import *

DT_CONTROL = 0.1  # s
LINEARIZE_STATE = [0, 0, 0, 3]
LINEARIZE_INPUT = [0, 0]


# We are looking for the input invariant set (maximal constraint admissible set)
# Use the (LQR) control law for the unconstrained system, since we define the terminal set to be a set
# within the state constraints.
# use algorithm 1 for `mpc_ex4.pdf` (`mpc_ex2.pdf` gives output admissible set)
# x(k + 1) = (A + BK)x(k)

# VERIFY IN THE END THAT: u within constraints
# DO SO BY: Kx = u --> x is constraint so u is also constraint

def compute_terminal_set(A_k: np.ndarray,
                         A_con: np.ndarray,
                         b_con: np.ndarray,
                         env: BaseEnv = None,
                         ) -> tuple[pc.Polytope, int]:
    """
    Obtain the terminal set that satisfies all the state constraints in the form (Ax <= b)
    :param A_k: A - BK; The A matrix when the optimal LQR control law is applied to the system
    :param A_con: matrix A in the constraints: Ax <= b
    :param b_con: matrix b in the constraints: Ax <= b
    :param env: the environment for which the terminal set is computed; contains the goal
    :return: terminal set (in the form Ax <= b) and the number of iterations necessary to obtain the solution
    """
    _, nx = A_k.shape
    s = len(A_con)
    goal = np.zeros(nx) if env is None else env.goal

    # Translate the input constraints to place the goal on the origin in order to get the terminal set around the goal
    p_in = pc.Polytope(A_con, b_con)
    p_in = p_in.translation(-np.array(env.goal))
    A_con, b_con = p_in.A, p_in.b

    for k in itertools.count(1):
        print(f"\n==================== k = {k} ==================== \n")
        # For all i = 1, 2, ..., s
        # Get the output of the constraint: A@x - b = ...
        #   Optimize x so that A@x - b is maximized -> if smaller than 0 all points are within this constraint
        # s.t. all constraints on x(0:k)
        for i in range(s):
            x = cp.Variable(nx)

            objective = cp.Maximize(A_con[i] @ (np.linalg.matrix_power(A_k, k + 1) @ x) - b_con[i])
            # range(k+1), bc also k
            A_term_set = np.vstack([A_con @ np.linalg.matrix_power(A_k, t) for t in range(k + 1)])
            b_term_set = np.hstack((b_con,) * (k + 1))
            # Constraints are build around the goal, so: A@(x - goal) <= b
            # in order to make the origin appear as the goal for the perspective of the constraints
            constraint = [A_term_set @ x - b_term_set <= 0]
            prob = cp.Problem(objective, constraint)
            result = prob.solve(verbose=False)
            print(x.value)
            if result <= 0 + 1e-3:
                print(f"True : {result}")
                if i == s - 1:
                    p = pc.Polytope(A_term_set, b_term_set)
                    # Translate the constraints back around the goal
                    p_goal = p.translation(goal)
                    return p_goal, k
            else:
                print(f"False : {result}")
                break

        if k > 100:
            print("Unsuccesful search: too many iterations")
            return None, None


def visualise_set(halfspaces: np.ndarray,
                  dims: Union[tuple[int, int], list[int]] = (0, 1),
                  extent: float = 10,
                  env: BaseEnv = None,
                  ) -> None:
    """
    Visualise 2 dimension of a set, e.g. the terminal set, by sampling a grid around a certain state (center)
    :param halfspaces:
    :param extent:
    :param dims:
    :param env: The environment for which the terminal set is computed
    :return:
    """
    center = [0, 0, 0, 0] if env is None else env.goal
    A_xy, b = halfspaces[..., dims], halfspaces[..., 4]

    ind = np.abs(A_xy.mean(axis=1)) > 0
    A_xy, b = A_xy[ind], b[ind]

    steps = 100
    off_x, off_y = center[dims[0]], center[dims[1]]
    xx, yy = np.meshgrid(np.linspace(-extent + off_x, extent + off_x, steps),
                         np.linspace(-extent + off_y, extent + off_y, steps))

    points = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = [xx[i, j], yy[i, j]]
            if np.all(A_xy @ point <= b).astype(bool):
                points.append(point)

    points = np.array(points)
    plt.figure()
    env.plot()
    plt.scatter(points[..., 0], points[..., 1], s=1, label='Inside set')
    plt.title(f"All region within the set for dimension: {dims}")
    plt.xlim(*env.lim[0])
    plt.ylim(*env.lim[1])
    plt.scatter(center[dims[0]], center[dims[1]], label='goal')
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.show()


if __name__ == "__main__":
    # Set up the environment
    environment = RoadMultipleCarsEnv()

    # Set up the MPC controller with horizon 1 to get the constraints
    controller = MPCStateFB(dt=DT_CONTROL, N=1, lin_state=LINEARIZE_STATE, lin_input=LINEARIZE_INPUT,
                            terminal_constraint=False,
                            input_constraint=False, state_constraint=True, env=environment)

    # Unconstrained optimal control law (LQR) : u = K x
    A, B, R, P = controller.A, controller.B, controller.R, controller.P
    K = - np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    # x+ = A x + B (K x) = (A + B K) x = A_k x
    A_k = A + B @ K

    # Print the system and control matrices
    print("============ Controller ============\n")
    print(f"A:\n{controller.A}\n")
    print(f"B:\n{controller.B}\n")
    print(f"K:\n{np.round(K, 2)}\n")
    print(f"A_k:\n{np.round(A_k, 2)}\n")

    # Get the state constraint matrices from the MPC controller
    assert controller.N == 1, "Horizon should be 1"
    A_con, b_con = controller.state_constraint()
    A_con = A_con[..., 4:]  # only use the constraints for x(1) (given that N=1)
    # Compute the controllability matrix
    controllability_matrix = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(5)])

    print("\n\n============ Constraints ============\n")
    print(f"A {A_con.shape}:\n{A_con}\n")
    print(f"b {b_con.shape}:\n{b_con}")
    print(f"\nThe rank of the controllability matrix is {np.linalg.matrix_rank(controllability_matrix)}",
          f"with n = {controller.nx}")

    # Calculate and plot the eigenvalues of:
    #   - The original system
    #   - The system with the LQR optimal control law
    eigenvalues_A = np.linalg.eig(A)[0]
    eigenvalues_A_k = np.linalg.eig(A_k)[0]
    # print(f"Eigenvalues A: {eigenvalues_A}")
    # print(f"\nEigenvalues Ak = A + BK: \n{eigenvalues_A_k}")

    x_A_k = [ele.real for ele in eigenvalues_A_k]  # extract real part
    y_A_k = [ele.imag for ele in eigenvalues_A_k]  # extract imaginary part
    x_A = [ele.real for ele in eigenvalues_A]  # extract real part
    y_A = [ele.imag for ele in eigenvalues_A]  # extract imaginary part

    # ======= UNCOMMENT LATER ========
    # plt.figure()
    # ax = plt.gca()
    # circle = plt.Circle((0, 0), 1, color=(0, 0, 0), fill=False, label='unit disk')
    # ax.add_patch(circle)
    # plt.scatter(x_A_k, y_A_k, label='eigenvalues A_k')
    # plt.scatter(x_A, y_A, label='eigenvalues A')
    # plt.title("Eigenvalues of original system and system with the LQR control law")
    # plt.grid()
    # ax.set_aspect('equal', 'box')
    # plt.legend()
    # plt.show()

    # Calculate the terminal set with the state constraints; does not respect the input constraints
    # Return as Ax - b <= 0; saved as halfspaces: [A, b]
    p, _ = compute_terminal_set(A_k, A_con, b_con, env=environment)
    p_red = pc.reduce(p)
    # A_term_set_red, b_term_set_red = p_red.A, p_red.b

    # We now the terminal set, which satisfy the state constraints, but not yet the input constraints.
    # LQR optimal control gain: u = Kx
    # Constraints on u (input): A_input_cnstr@u <= b_input_cnstr  -->  A_input_cnstr@K@x <= b_input_cnstr
    # So additional constraints on x --> (A_input_cnstr@K)@x <= b_input_cnstr
    A_input, b_input = controller.input_constraint()
    p_input = pc.Polytope(A_input @ K, b_input)
    p_input = p_input.translation(environment.goal)

    p_terminal_set = p_red.intersect(p_input)

    A_term_set_final, b_term_set_final = p_terminal_set.A, p_terminal_set.b
    halfspaces = np.hstack((A_term_set_final, np.expand_dims(b_term_set_final, axis=1)))

    # ===== To save the set to a file; name is based on: environment name + goal state =====
    filename = environment.name + ''.join('_' + str(s) for s in environment.goal) + '.npy'
    print(f"Filename to save: {filename}")
    # np.save('../terminal_sets/' + filename, halfspaces)
    print(f"Terminal set has {len(halfspaces)} constraints.")

    # Visualize the terminal set
    visualise_set(halfspaces, extent=2, env=environment)

    print(f"Terminal set:\n{p_red}")

