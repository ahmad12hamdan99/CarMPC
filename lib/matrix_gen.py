import numpy as np

from lib.simulator import CarTrailerDimension


def predmod(A: np.ndarray, B: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    This functions creates the matrices T and S in the following equation:
    state space model: x(k+1) = A x(k) + B u(k) with mpc horizon N
    x_ = the sequence {x(0), x(1), ..., x(N)}
    x0 = x(0)
    u_ = the sequence {u(0), u(1), ..., u(N-1)}

    x_ = T x0 + S u_

    T = [I; A; A^{2}; ...; A^{N-1}]
    S = [0        0        ...   0]
        [B        0        ...   0]
        [AB       B        ...   0]
        [:        :        ...   0]
        [A^{N-1}B A^{N-2}B ...   B]
    :param A: is matrix A in the ss model
    :param B: is matrix B in the ss model
    :param N: is the mpc horizon
    :return: matrices T and S
    """
    T = np.vstack([np.linalg.matrix_power(A, i) for i in range(N + 1)])
    S = np.hstack(([np.vstack(
        [np.linalg.matrix_power(A, i - j - 1) @ B if i - j - 1 >= 0 else np.zeros_like(B) for i in range(N + 1)]) for j
        in range(N)]))

    return T, S


def costgen(Q: np.ndarray, R: np.ndarray, P: np.ndarray, T: np.ndarray, S: np.ndarray, nx: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to calculate the matrices H and h and the constant term in the following cost equation with terminal cost:
    V_N = 1/2 [ Sum{k=0}{N-1} x(k)^T Q x(k) + u(k)^T R u(k) ]  + x(N)^T P x(N)  [1]

    To the following matrix equation:
    V_N = 1/2 u_^T H u_ + h.T x0 u_ + 1/2 x0^T constant x0     [2]

    Where:
    H = S^T Q S + R
    h = T^T Q S
    constant = x0^T T^T Q T x0

    x_ = the sequence {x(0), x(1), ..., x(N)}
    x0 = x(0)
    u_ = the sequence {u(0), u(1), ..., u(N-1)}

    :param Q: the Q matrix from equation [1]
    :param R: the R matrix from equation [1]
    :param P: the P matrix from equation [1]
    :param T: the T matrix from the function predmod
    :param S: the S matrix from the function predmod
    :param nx: the state dimension
    :return: H, h, constant
    """
    N = len(S) // nx - 1

    Q_hat = stack_matrix_along_diag(Q, N+1)
    Q_hat[N*nx:(N+1)*nx, N*nx:(N+1)*nx] = P

    R_hat = stack_matrix_along_diag(R, N)

    H = R_hat +  S.T @ Q_hat @ S
    h = S.T @ Q_hat @ T
    constant = T.T @ Q_hat @ T

    return H, h, constant


def stack_matrix_along_diag(A: np.ndarray, N: int) -> np.ndarray:
    i, j = A.shape
    A_hat = np.zeros((i * N, j * N))

    for n in range(N):
        A_hat[i * n:i * (n + 1), j * n:j * (n + 1)] = A

    return A_hat
