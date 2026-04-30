import numpy as np
from helpers import get_rmse, get_ndcg, get_norm, update_best_sofar
from als_update import als_update, als_update_no_UV, als_update_UV_only


def fit_model_no_UV(Y_train, Y_csr, Y_csc, mu, b, c, l_b, df_valid, thresh, info=2):
    assert info in (0, 1, 2)
    preds_rmse = np.clip(mu + b[df_valid.rows] + c[df_valid.cols], 1, 10)
    preds_ndcg = mu + b[:, None] + c[None, :]
    valid_rmse_old = get_rmse(preds_rmse, df_valid)
    valid_ndcg_old = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
    if info == 2:
        print(
            "original:               valid rmse =",
            f"{valid_rmse_old:.6f}" + ",",
            "valid ndcg =",
            f"{valid_ndcg_old:.6f}",
        )

    max_norm_diff_old = None
    valid_rmse_diff_old = None
    valid_ndcg_diff_old = None
    counter = 1
    while True:
        mu_old, b_old, c_old = mu.copy(), b.copy(), c.copy()
        mu, b, c = als_update_no_UV(Y_csr, Y_csc, mu, b, c, l_b)

        mu_norm = get_norm(mu_old, mu)
        b_norm, c_norm = get_norm(b_old, b), get_norm(c_old, c)
        max_norm_diff = max(mu_norm, b_norm, c_norm)

        preds_rmse = np.clip(mu + b[df_valid.rows] + c[df_valid.cols], 1, 10)
        preds_ndcg = mu + b[:, None] + c[None, :]
        valid_rmse = get_rmse(preds_rmse, df_valid)
        valid_ndcg = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
        valid_rmse_diff = valid_rmse_old - valid_rmse
        valid_ndcg_diff = valid_ndcg_old - valid_ndcg

        if info == 2:
            print(
                f"         {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}, {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}"
            )

        if valid_rmse_diff < thresh:
            if valid_rmse_diff < 0:
                if info == 2:
                    print(
                        "valid_rmse_diff is negative, returning the last iteration's values"
                    )
                counter -= 1
                max_norm_diff = max_norm_diff_old
                valid_rmse_diff = valid_rmse_diff_old
                valid_rmse = valid_rmse_old
                valid_ndcg_diff = valid_ndcg_diff_old
                valid_ndcg = valid_ndcg_old
                res = mu_old, b_old, c_old
            else:
                res = mu, b, c

            if info > 0:
                print(
                    f"results: {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}, {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}"
                )
            return res

        counter += 1
        valid_rmse_old = valid_rmse
        valid_ndcg_old = valid_ndcg
        max_norm_diff_old = max_norm_diff
        valid_rmse_diff_old = valid_rmse_diff
        valid_ndcg_diff_old = valid_ndcg_diff


def fit_model_full(
    Y_train, Y_csr, Y_csc, mu, b, c, U, V, l_b, l_f, df_valid, thresh, info=1
):
    assert info in (0, 1, 2)

    biases = mu + b[df_valid.rows] + c[df_valid.cols]
    interactions = np.sum(U[df_valid.rows] * V[df_valid.cols], axis=1)
    preds_rmse = np.clip(biases + interactions, 1, 10)
    preds_ndcg = mu + b[:, None] + c[None, :] + U @ V.T
    valid_rmse_old = get_rmse(preds_rmse, df_valid)
    valid_ndcg_old = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
    if info == 2:
        print(
            "original:               valid rmse =",
            f"{valid_rmse_old:.6f}" + ",",
            "valid ndcg =",
            f"{valid_ndcg_old:.6f}",
        )

    max_norm_diff_old = None
    valid_rmse_diff_old = None
    valid_ndcg_diff_old = None
    counter = 1
    while True:
        mu_old, b_old, c_old = mu.copy(), b.copy(), c.copy()
        U_old, V_old = U.copy(), V.copy()
        mu, b, c, U, V = als_update(Y_csr, Y_csc, mu, b, c, U, V, l_b, l_f)

        mu_norm = get_norm(mu_old, mu)
        b_norm, c_norm = get_norm(b_old, b), get_norm(c_old, c)
        U_norm, V_norm = get_norm(U_old, U), get_norm(V_old, V)
        max_norm_diff = max(mu_norm, b_norm, c_norm, U_norm, V_norm)

        biases = mu + b[df_valid.rows] + c[df_valid.cols]
        interactions = np.sum(U[df_valid.rows] * V[df_valid.cols], axis=1)
        preds_rmse = np.clip(biases + interactions, 1, 10)
        preds_ndcg = mu + b[:, None] + c[None, :] + U @ V.T
        valid_rmse = get_rmse(preds_rmse, df_valid)
        valid_ndcg = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
        valid_rmse_diff = valid_rmse_old - valid_rmse
        valid_ndcg_diff = valid_ndcg - valid_ndcg_old

        if info == 2:
            print(
                f"         {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}"  # , {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}
            )

        if valid_rmse_diff < thresh:
            if valid_rmse_diff < 0:
                if info == 2:
                    print(
                        "valid_rmse_diff is negative, returning the last iteration's values"
                    )
                counter -= 1
                max_norm_diff = max_norm_diff_old
                valid_rmse_diff = valid_rmse_diff_old
                valid_rmse = valid_rmse_old
                valid_ndcg_diff = valid_ndcg_diff_old
                valid_ndcg = valid_ndcg_old
                res = mu_old, b_old, c_old, U_old, V_old
            else:
                res = mu, b, c, U, V

            if info > 0:
                print(
                    f"results: {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}"  # , {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}
                )
            return res

        counter += 1
        valid_rmse_old = valid_rmse
        valid_ndcg_old = valid_ndcg
        max_norm_diff_old = max_norm_diff
        valid_rmse_diff_old = valid_rmse_diff
        valid_ndcg_diff_old = valid_ndcg_diff


def fit_model_full_beta(
    Y_train, Y_csr, Y_csc, mu, b, c, U, V, l_b, l_f, beta, df_valid, thresh, info=1
):
    assert info in (0, 1, 2)

    biases = mu + b[df_valid.rows] + c[df_valid.cols]
    preds_rmse = np.clip(
        biases + np.sum(U[df_valid.rows] * V[df_valid.cols], axis=1), 1, 10
    )
    preds_ndcg = mu + b[:, None] + beta * c[None, :] + U @ V.T
    valid_rmse_old = get_rmse(preds_rmse, df_valid)
    valid_ndcg_old = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
    if info == 2:
        print(
            "original:     valid rmse =",
            f"{valid_rmse_old:.6f}" + ",",
            "valid ndcg =",
            f"{valid_ndcg_old:.6f}",
        )

    best_sofar = update_best_sofar(
        {}, mu, b, c, U, V, valid_rmse_old, valid_ndcg_old, None, None, None, None
    )

    best_ndcg_for_patience = valid_ndcg_old

    counter = 1
    patience = 0
    while True:
        mu_old, b_old, c_old = mu.copy(), b.copy(), c.copy()
        U_old, V_old = U.copy(), V.copy()
        mu, b, c, U, V = als_update(Y_csr, Y_csc, mu, b, c, U, V, l_b, l_f)

        mu_norm = get_norm(mu_old, mu)
        b_norm, c_norm = get_norm(b_old, b), get_norm(c_old, c)
        U_norm, V_norm = get_norm(U_old, U), get_norm(V_old, V)
        max_norm_diff = max(mu_norm, b_norm, c_norm, U_norm, V_norm)

        biases = mu + b[df_valid.rows] + c[df_valid.cols]
        interactions = np.sum(U[df_valid.rows] * V[df_valid.cols], axis=1)
        preds_rmse = np.clip(biases + interactions, 1, 10)
        preds_ndcg = mu + b[:, None] + beta * c[None, :] + U @ V.T
        valid_rmse = get_rmse(preds_rmse, df_valid)
        valid_ndcg = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
        valid_rmse_diff = best_sofar["valid_rmse"] - valid_rmse
        valid_ndcg_diff = valid_ndcg - best_sofar["valid_ndcg"]

        if info == 2:
            print(
                f"         {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}"  # , {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}
            )

        if valid_ndcg > best_sofar["valid_ndcg"]:
            best_sofar = update_best_sofar(
                best_sofar,
                mu,
                b,
                c,
                U,
                V,
                valid_rmse,
                valid_ndcg,
                counter,
                max_norm_diff,
                valid_rmse_diff,
                valid_ndcg_diff,
            )

        if valid_ndcg > best_ndcg_for_patience + thresh:
            best_ndcg_for_patience = valid_ndcg
            if patience > 0 and info == 2:
                print("         patience restored")
            patience = 0

        else:
            patience += 1
            if info == 2:
                print(f"         valid_ndcg_diff is under thresh, {patience=}")

            if patience >= 3:
                max_norm_diff = best_sofar["max_norm_diff"]
                valid_rmse_diff = best_sofar["valid_rmse_diff"]
                valid_rmse = best_sofar["valid_rmse"]
                valid_ndcg_diff = best_sofar["valid_ndcg_diff"]
                valid_ndcg = best_sofar["valid_ndcg"]
                counter = best_sofar["counter"]

                if info > 0:
                    print(
                        f"results: {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}"  # , {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}
                    )
                return (
                    best_sofar["mu"],
                    best_sofar["b"],
                    best_sofar["c"],
                    best_sofar["U"],
                    best_sofar["V"],
                )

        counter += 1


def fit_model_UV_only(Y_train, Y_csr, Y_csc, U, V, l_f, df_valid, thresh, info=1):
    assert info in (0, 1, 2)

    preds_rmse = np.clip(np.sum(U[df_valid.rows] * V[df_valid.cols], axis=1), 1, 10)
    preds_ndcg = U @ V.T
    valid_rmse_old = get_rmse(preds_rmse, df_valid)
    valid_ndcg_old = get_ndcg(preds_ndcg, Y_train, df_valid, 5)
    if info == 2:
        print(
            "original:                  valid rmse =",
            f"{valid_rmse_old:.6f}" + ",",
            "valid ndcg =",
            f"{valid_ndcg_old:.6f}",
        )

    best_sofar = {
        "U": U.copy(),
        "V": V.copy(),
        "valid_rmse": valid_rmse_old,
        "valid_ndcg": valid_ndcg_old,
        "counter": 0,
        "max_norm_diff": np.nan,
    }

    best_ndcg_for_patience = valid_ndcg_old

    counter = 1
    patience = 0
    while True:
        U_old, V_old = U.copy(), V.copy()
        U, V = als_update_UV_only(Y_csr, Y_csc, U, V, l_f)

        U_norm, V_norm = get_norm(U_old, U), get_norm(V_old, V)
        max_norm_diff = max(U_norm, V_norm)

        preds_rmse = np.clip(np.sum(U[df_valid.rows] * V[df_valid.cols], axis=1), 1, 10)
        preds_ndcg = U @ V.T
        valid_rmse = get_rmse(preds_rmse, df_valid)
        valid_ndcg = get_ndcg(preds_ndcg, Y_train, df_valid, 5)

        if info == 2:
            print(
                f"         {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}"  # , {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}
            )

        if valid_ndcg > best_sofar["valid_ndcg"]:
            best_sofar["U"] = U.copy()
            best_sofar["V"] = V.copy()
            best_sofar["valid_rmse"] = valid_rmse
            best_sofar["valid_ndcg"] = valid_ndcg
            best_sofar["counter"] = counter
            best_sofar["max_norm_diff"] = max_norm_diff

        if valid_ndcg > best_ndcg_for_patience + thresh:
            best_ndcg_for_patience = valid_ndcg
            if patience > 0 and info == 2:
                print(23 * " ", "patience restored")
            patience = 0

        else:
            patience += 1
            if info == 2:
                print(23 * " ", f"valid_ndcg_diff is under thresh, {patience=}")

            if patience >= 3:
                max_norm_diff = best_sofar["max_norm_diff"]
                valid_rmse = best_sofar["valid_rmse"]
                valid_ndcg = best_sofar["valid_ndcg"]
                counter = best_sofar["counter"]

                if info > 0:
                    print(
                        f"results: {counter = :>3}, {max_norm_diff = :>7.2f}, {valid_rmse = :<8.6f}, {valid_ndcg = :<8.6f}"  # , {valid_rmse_diff = :<7.2g}, {valid_ndcg_diff = :<7.2g}
                    )
                return best_sofar["U"], best_sofar["V"]

        counter += 1


def infer_user_vector(V, item_ids, ratings, lambda_fact):
    k = V.shape[1]
    I = np.eye(k)

    V_u = V[item_ids, :]  # shape (n_ratings, k)
    r_u = ratings  # shape (n_ratings,)

    A = V_u.T @ V_u + lambda_fact * I
    rhs = V_u.T @ r_u

    u = np.linalg.solve(A, rhs)
    return u
