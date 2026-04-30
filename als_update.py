import numpy as np


def get_inds_vals(Y_sparse, ind):
    start, end = Y_sparse.indptr[ind], Y_sparse.indptr[ind + 1]
    inds = Y_sparse.indices[start:end]
    vals = Y_sparse.data[start:end]
    return inds, vals


def update_mu(Y_csr, b, c, U, V, no_UV=False):
    total = 0
    for a in range(Y_csr.shape[0]):
        books, vals = get_inds_vals(Y_csr, a)

        if vals.size == 0:
            continue

        if no_UV:
            total += np.sum(vals - b[a] - c[books])
        else:
            total += np.sum(vals - b[a] - c[books] - V[books, :] @ U[a, :])

    mu = total / Y_csr.nnz
    return mu


def update_b(Y_csr, mu, c, U, V, lam, no_UV=False):
    n = Y_csr.shape[0]
    b = np.zeros(n)

    for a in range(n):
        books, vals = get_inds_vals(Y_csr, a)

        if vals.size == 0:
            continue

        if no_UV:
            residuals = vals - mu - c[books]
        else:
            residuals = vals - mu - c[books] - V[books, :] @ U[a, :]

        b[a] = (residuals).sum() / (vals.size + lam)

    return b


def update_c(Y_csc, mu, b, U, V, lam, no_UV=False):
    m = Y_csc.shape[1]
    c = np.zeros(m)

    for i in range(m):
        users, vals = get_inds_vals(Y_csc, i)

        if vals.size == 0:
            continue

        if no_UV:
            residuals = vals - mu - b[users]
        else:
            residuals = vals - mu - b[users] - U[users, :] @ V[i, :]
        c[i] = (residuals).sum() / (vals.size + lam)

    return c


def update_U(Y_csr, mu, b, c, V, lam):
    n = Y_csr.shape[0]
    k = V.shape[1]

    I = np.eye(k)
    U = np.zeros((n, k))

    for a in range(n):
        books, vals = get_inds_vals(Y_csr, a)

        if vals.size == 0:
            continue

        V_a = V[books, :]
        target = vals - mu - b[a] - c[books]

        A = V_a.T @ V_a + lam * I
        rhs = V_a.T @ target

        U[a, :] = np.linalg.solve(A, rhs)

    return U


def update_V(Y_csc, mu, b, c, U, lam):
    m = Y_csc.shape[1]
    k = U.shape[1]

    I = np.eye(k)
    V = np.zeros((m, k))

    for i in range(m):
        users, vals = get_inds_vals(Y_csc, i)

        if vals.size == 0:
            continue

        U_i = U[users, :]
        target = vals - mu - b[users] - c[i]

        A = U_i.T @ U_i + lam * I
        rhs = U_i.T @ target

        V[i, :] = np.linalg.solve(A, rhs)

    return V


def als_update(Y_csr, Y_csc, mu, b, c, U, V, lambda_bias, lambda_fact):
    """Performs one ALS update"""
    mu = update_mu(Y_csr, b, c, U, V)
    b = update_b(Y_csr, mu, c, U, V, lambda_bias)
    c = update_c(Y_csc, mu, b, U, V, lambda_bias)
    U = update_U(Y_csr, mu, b, c, V, lambda_fact)
    V = update_V(Y_csc, mu, b, c, U, lambda_fact)

    return mu, b, c, U, V


def als_update_no_UV(Y_csr, Y_csc, mu, b, c, lambda_bias):
    """Performs one ALS update; model without U and V interactions"""
    mu = update_mu(Y_csr, b, c, None, None, True)
    b = update_b(Y_csr, mu, c, None, None, lambda_bias, True)
    c = update_c(Y_csc, mu, b, None, None, lambda_bias, True)

    return mu, b, c


def als_update_UV_only(Y_csr, Y_csc, U, V, lambda_fact):
    """Performs one ALS update on a model with only UV interactions"""
    n = Y_csr.shape[0]
    m = Y_csc.shape[1]
    k = V.shape[1]

    def update_U(Y_csr, V, lam):
        I = np.eye(k)
        U = np.zeros((n, k))
        for a in range(n):
            books, vals = get_inds_vals(Y_csr, a)
            if vals.size == 0:
                continue
            V_a = V[books, :]
            A = V_a.T @ V_a + lam * I
            rhs = V_a.T @ vals
            U[a, :] = np.linalg.solve(A, rhs)
        return U

    def update_V(Y_csc, U, lam):
        I = np.eye(k)
        V = np.zeros((m, k))
        for i in range(m):
            users, vals = get_inds_vals(Y_csc, i)
            if vals.size == 0:
                continue
            U_i = U[users, :]
            A = U_i.T @ U_i + lam * I
            rhs = U_i.T @ vals
            V[i, :] = np.linalg.solve(A, rhs)
        return V

    U = update_U(Y_csr, V, lambda_fact)
    V = update_V(Y_csc, U, lambda_fact)

    return U, V


# dense version (just for mu, b, c)

# UV = U @ V.T
# mu_dense = ((Y_train - UV - b[:, None] - c[None, :]) * O_mask).sum() / O_size
# b_dense = ((Y_train - mu_dense - c[None, :] - UV) * O_mask).sum(axis=1) / (
#     O_mask.sum(axis=1) + lambda_bias
# )
# c_dense = ((Y_train - mu_dense - b_dense[:, None] - UV) * O_mask).sum(axis=0) / (
#     O_mask.sum(axis=0) + lambda_bias
# )

# print("norm:", round(abs(mu - mu_dense), 6))
# print("norm:", round(np.linalg.norm(b - b_dense), 6))
# print("norm:", round(np.linalg.norm(c - c_dense), 6))
