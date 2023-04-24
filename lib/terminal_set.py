import numpy as np
import itertools
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.patches as patches
import polytope as pc
from typing import Union

from lib.environments import BaseEnv
from lib.mpc import MPCStateFB
from lib.configuration import *


# We are looking for the input invariant set (maximal constraint admissible set)
# Use the (LQR) control law for the unconstrained system, since we define the terminal set to be a set
# within the state constraints.
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
            if result <= 0 + 1e-3:
                if i == s - 1:
                    p = pc.Polytope(A_term_set, b_term_set)
                    # Translate the constraints back around the goal
                    p_goal = p.translation(goal)
                    return p_goal, k
            else:
                break

        if k > 100:
            print("Unsuccesful search: too many iterations. The goal might be outside the constraints.")
            return None, None


def visualise_set(A_constraints: np.ndarray,
                  b_constraints: np.ndarray,
                  env: BaseEnv,
                  extent: float = 25,
                  ) -> None:
    """
    Visualise a set, e.g. the terminal set, by projecting it on the x,y-plane.
    A grid is sampled around a certain state (environment goal otherwise origin).
    Set should be represented by:
        A x + b <= 0
    :param A_constraints: the A matrix of the set constraints
    :param b_constraints: the b matrix of the set constraints
    :param env: The environment for which the terminal set is computed
    :param extent: The range around the center point to sample
    :return:
    """
    center = [0, 0, 0, 0] if env is None else env.goal
    halfspaces = np.hstack((A_constraints, np.expand_dims(b_constraints, axis=1)))
    A, b = halfspaces[..., :4], halfspaces[..., 4]

    steps = 100
    off_x, off_y = center[0], center[1]
    xx, yy = np.meshgrid(np.linspace(-extent + off_x, extent + off_x, steps),
                         np.linspace(-extent + off_y, extent + off_y, steps))
    plt.figure()
    ax = plt.gca()

    blue = np.array([17, 73, 112]) / 255
    # The range of psi and v
    psi = 0
    v_range = np.arange(6)
    for idx, v in enumerate(v_range):
        points = []
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                point = [xx[i, j], yy[i, j], psi, v]
                if np.all(A @ point <= b).astype(bool):
                    points.append(point)

        if len(points) != 0:
            points = np.array(points)
            hull = pc.qhull(points[..., :2])
            hull.plot(ax=ax, color=blue + (idx + 1) * 0.5 / len(v_range), linestyle='solid', edgecolor='none')

    env.plot()
    # plt.scatter(points[..., 0], points[..., 1], s=1, label='Inside set')
    plt.title(f"The terminal set projected on the x,y-plane \n for v = {v_range} m/s and " + r'$\psi$' + " = 0 rad")
    plt.xlim(*env.lim[0])
    plt.ylim(*env.lim[1])
    handles, _ = ax.get_legend_handles_labels()
    term_set = patches.Patch(color=blue + 0.25, label='Terminal set')
    print(handles)
    ax.legend(loc='upper center', handles=handles + env.handles + [term_set],
              bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3)
    ax.set_aspect('equal', 'box')
    plt.show()


def calc_terminal_set(env: BaseEnv, verbose: bool = False, save: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the terminal set for a certain environment.
    Be aware that the goal position influences the position and shape of the terminal set.
    :param env: environment for which the terminal set should be calculated
    :param verbose: Print all the matrices of the controller and plot the eigenvalues
    :param save: save the terminal set to a file
    :return:
    """
    # Set up the MPC controller with horizon 1 to get the constraints
    controller = MPCStateFB(dt=DT_CONTROL, N=1, lin_state=LINEARIZE_STATE, lin_input=LINEARIZE_INPUT,
                            terminal_constraint=False,
                            input_constraint=False, state_constraint=True, env=env)
    # A_k = A + B K
    A_k = controller.A + controller.B @ controller.K

    if verbose:
        # Print the system and control matrices
        print("============ Controller ============\n")
        print(f"A:\n{controller.A}\n")
        print(f"B:\n{controller.B}\n")
        print(f"K:\n{np.round(controller.K, 2)}\n")
        print(f"A_k:\n{np.round(A_k, 2)}\n")

    # Get the state constraint matrices from the MPC controller
    A_con, b_con = controller.state_constraint()
    A_con = A_con[..., 4:]  # only use the constraints for x(1) (given that N=1)

    if verbose:
        # Compute the controllability matrix
        controllability_matrix = np.hstack([np.linalg.matrix_power(controller.A, i) @ controller.B for i in range(4)])
        print("\n\n============ Constraints ============\n")
        print(f"A {A_con.shape}:\n{A_con}\n")
        print(f"b {b_con.shape}:\n{b_con}")
        print(f"\nThe rank of the controllability matrix is {np.linalg.matrix_rank(controllability_matrix)}",
              f"with n = {controller.nx}")

        # Plot eigenvalues of original system and system with unconstrained control law K
        eigenvalues_A = np.linalg.eig(controller.A)[0]
        eigenvalues_A_k = np.linalg.eig(A_k)[0]

        x_A_k = [ele.real for ele in eigenvalues_A_k]  # extract real part
        y_A_k = [ele.imag for ele in eigenvalues_A_k]  # extract imaginary part
        x_A = [ele.real for ele in eigenvalues_A]  # extract real part
        y_A = [ele.imag for ele in eigenvalues_A]  # extract imaginary part

        plt.figure()
        ax = plt.gca()
        circle = plt.Circle((0, 0), 1, color=(0, 0, 0), fill=False, label='unit disk')
        ax.add_patch(circle)
        plt.scatter(x_A_k, y_A_k, label='eigenvalues A_k')
        plt.scatter(x_A, y_A, label='eigenvalues A')
        plt.title("Eigenvalues of original system and system with the LQR control law")
        plt.grid()
        ax.set_aspect('equal', 'box')
        plt.legend()
        plt.show()

    # Calculate the invariant set (Does not respect the input constraints)
    # Return as Ax - b <= 0; saved as a polytope object
    p, _ = compute_terminal_set(A_k, A_con, b_con, env=env)  # x+ \in X_f given x \in X_f for u \in R^2

    # Terminal set also needs to respect the input constraints
    A_input, b_input = controller.input_constraint()
    p_input = pc.Polytope(A_input @ controller.K, b_input)  # All u \in U given U = Kx
    p_input = p_input.translation(env.goal)

    # Take the intersection of both sets and obtain the terminal set
    p_terminal_set = pc.reduce(p.intersect(p_input))  # x+ \in X_f given x \in X_f and u \in U where u = Kx

    # ===== To save the set to a file; name is based on: environment name + goal state =====
    A_term_set_final, b_term_set_final = p_terminal_set.A, p_terminal_set.b
    halfspaces = np.hstack((A_term_set_final, np.expand_dims(b_term_set_final, axis=1)))
    filename = env.name + ''.join('_' + str(s) for s in env.goal) + '.npy'
    print(f"Filename to save: {filename}")
    np.save('../terminal_sets/' + filename, halfspaces)
    print(f"Terminal set has {len(halfspaces)} constraints.")

    return A_term_set_final, b_term_set_final
