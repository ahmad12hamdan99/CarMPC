import numpy as np


def algorithm_1(G: np.ndarray, H: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    n = the state dimension
    m = the input dimension = 1
    :param G: R^{s x n}
    :param H: R^{s x m} = R^{s x 1}
    :param phi: R^{s x 1}
    :return: P and gamma
    """
    # PART 1
    s, n = G.shape
    I_0 = np.where(H == 0)[0]
    I_plus = np.where(H > 0)[0]
    I_min = np.where(H < 0)[0]
    s_0 = len(I_0)
    s_plus = len(I_plus)
    s_min = len(I_min)
    assert s == s_0 + s_plus + s_min

    # PART 2
    C = np.concatenate((G, np.expand_dims(phi, axis=1)), axis=1)
    r = s_0 + s_plus * s_min

    # print(I_0, I_plus, I_min)
    D = np.zeros((r, n + 1))
    D = []
    for i in I_0:
        D.append(C[i])

    for i in I_plus:
        for j in I_min:
            D.append(H[i] * C[j] - H[j] * C[i])

    D = np.array(D)
    P = D[:, :-1]
    gamma = D[:, -1]
    return P, gamma


def algorithm_2(G: np.ndarray, H: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    n = the state dimension
    m = the input dimension

    Where:
        G x0 + H u + phi <= 0

    Where:
        P x + gamma <= 0
        This gives a convex area wherein x satisfies all the constraints

    :param G: R^{s x n}
    :param H: R^{s x m}
    :param phi: R^{s x 1}
    :return: P and gamma
    """
    s, n = G.shape
    m = H.shape[1] if len(H.shape) >= 2 else 1
    assert m > 1, "Use algorithm_1"

    j = m
    G_j = np.concatenate((G, H[:, :m - 1]), axis=1)
    H_j = H[:, m - 1]
    phi_j = phi

    while True:
        if j == 0:
            return P_j, gamma_j
        else:
            P_j, gamma_j = algorithm_1(G_j, H_j, phi_j)
            j -= 1
            G_j = P_j[:, :-1]
            H_j = P_j[:, -1]
            phi_j = gamma_j
