import numpy as np
import pandas as pd


def valid_test_select(Y, n_valid, n_test, seed=123):

    rows, cols = np.nonzero(Y)
    rng = np.random.default_rng(seed=seed)

    perm = rng.permutation(len(rows))
    valid_sel = perm[:n_valid]
    test_sel = perm[n_valid : n_valid + n_test]

    valid_rows, valid_cols = rows[valid_sel], cols[valid_sel]
    test_rows, test_cols = rows[test_sel], cols[test_sel]

    valid_vals = Y[valid_rows, valid_cols]
    test_vals = Y[test_rows, test_cols]

    Y_train = Y.copy()
    Y_train[valid_rows, valid_cols] = 0
    Y_train[test_rows, test_cols] = 0

    valid_data_arr = np.stack([valid_rows, valid_cols, valid_vals], axis=1)
    test_data_arr = np.stack([test_rows, test_cols, test_vals], axis=1)

    valid_data = pd.DataFrame(valid_data_arr, columns=["rows", "cols", "vals"])
    test_data = pd.DataFrame(test_data_arr, columns=["rows", "cols", "vals"])

    return Y_train, valid_data, test_data


def valid_test_select_per_user(Y, include_test=True, seed=123):
    rng = np.random.default_rng(seed=seed)

    def split_counts(n):
        if n == 5:
            n_valid, n_test = (1, 1) if include_test else (2, 0)
        if n <= 7:
            n_valid, n_test = (2, 1) if include_test else (2, 0)
        elif n <= 12:
            n_valid, n_test = (3, 1) if include_test else (3, 0)
        elif n <= 20:
            n_valid, n_test = (4, 2) if include_test else (5, 0)
        else:
            n_valid, n_test = (4, 2) if include_test else (6, 0)

        n_train = n - n_valid - n_test
        return n_train, n_valid, n_test

    valid_rows = []
    valid_cols = []
    test_rows = []
    test_cols = []

    Y_train = Y.copy()

    for u in range(Y.shape[0]):
        user_cols = np.nonzero(Y[u])[0]
        n = len(user_cols)

        if n < 5:
            raise ValueError(f"User {u} has only {n} ratings; expected at least 5.")

        n_train, n_valid, n_test = split_counts(n)
        perm = rng.permutation(n)

        valid_sel = perm[n_train : n_train + n_valid]
        valid_cols_u = user_cols[valid_sel]
        valid_rows.extend([u] * n_valid)
        valid_cols.extend(valid_cols_u)

        if include_test:
            test_sel = perm[n_train + n_valid : n_train + n_valid + n_test]
            test_cols_u = user_cols[test_sel]
            test_rows.extend([u] * n_test)
            test_cols.extend(test_cols_u)

    valid_rows = np.asarray(valid_rows)
    valid_cols = np.asarray(valid_cols)

    if include_test:
        test_rows = np.asarray(test_rows)
        test_cols = np.asarray(test_cols)

    valid_vals = Y[valid_rows, valid_cols]
    Y_train[valid_rows, valid_cols] = 0
    valid_data_arr = np.stack([valid_rows, valid_cols, valid_vals], axis=1)
    valid_data = pd.DataFrame(valid_data_arr, columns=["rows", "cols", "vals"])

    if include_test:
        test_vals = Y[test_rows, test_cols]
        Y_train[test_rows, test_cols] = 0
        test_data_arr = np.stack([test_rows, test_cols, test_vals], axis=1)
        test_data = pd.DataFrame(test_data_arr, columns=["rows", "cols", "vals"])

    if include_test:
        return Y_train, valid_data, test_data  # type: ignore
    else:
        return Y_train, valid_data, None
