import numpy as np


def get_rmse(preds, df):
    return float(np.sqrt(np.mean((df.vals - preds) ** 2)))


def get_norm(old, new):
    return float(np.linalg.norm(old - new))


def get_isbn(pos, book_cats, Y_columns):
    """Given position of column in matrix Y, return the original ISBN."""
    return book_cats.cat.categories[Y_columns[pos]]


def get_Ypos(isbn, book_cats, Y_columns):
    """Given ISBN, return the column position in matrix Y, or None if missing."""
    try:
        code = book_cats.cat.categories.get_loc(isbn)
    except KeyError:
        return None

    matches = np.where(Y_columns == code)[0]
    return matches[0] if len(matches) > 0 else None


def get_book_info(isbn, filepath="Books.csv"):
    """
    Find first row starting with given ISBN and return:
    everything after first comma and before first ',http://'
    """
    isbn = str(isbn)

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(isbn + ","):
                # remove ISBN + comma
                rest = line[len(isbn) + 1 :]

                # cut before first ",http://"
                split_idx = rest.find(",http://")
                if split_idx != -1:
                    return rest[:split_idx]
                else:
                    return rest.strip()


def get_ndcg(Y_pred, Y_train, valid_data, K):
    _, n_items = Y_pred.shape

    scores = Y_pred.copy()
    scores[Y_train > 0] = -1e18  # exclude already-seen training items

    discounts = np.log2(np.arange(2, K + 2))
    ndcgs = []

    for u, g in valid_data.groupby("rows"):
        u = int(u)

        y_true = np.zeros(n_items, dtype=float)

        cols = g["cols"].to_numpy(dtype=int)
        vals = g["vals"].to_numpy(dtype=float)
        y_true[cols] = vals

        # get top-K without full sorting: O(n_items)
        topk = np.argpartition(-scores[u], K - 1)[:K]

        # sort only the selected top-K items
        topk = topk[np.argsort(-scores[u, topk])]

        # DCG@5
        gains = 2 ** y_true[topk] - 1
        dcg = np.sum(gains / discounts)

        # IDCG@5: best possible validation items
        ideal_topk = np.argpartition(-y_true, K - 1)[:K]
        ideal_topk = ideal_topk[np.argsort(-y_true[ideal_topk])]

        ideal_gains = 2 ** y_true[ideal_topk] - 1
        idcg = np.sum(ideal_gains / discounts)

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else np.nan


def recommend(Y: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k values per row of Y.

    Ordering is:
        1. by Y value (descending)
        2. ties broken by c (descending)

    Parameters
    ----------
    Y : np.ndarray
        2D array of shape (n_rows, n_cols)
    c : np.ndarray
        1D array of shape (n_cols,)
    k : int
        Number of top elements to return per row

    Returns
    -------
    np.ndarray
        Array of shape (n_rows, k) containing column indices.
        Each row is sorted from best to worst according to the rule.
    """
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array.")
    if c.ndim != 1:
        raise ValueError("c must be a 1D array.")
    if Y.shape[1] != c.shape[0]:
        raise ValueError("c must have same length as number of columns in Y.")
    if k <= 0 or k > Y.shape[1]:
        raise ValueError("k must be in range 1..n_cols.")

    # Broadcast c to match Y shape
    c_broadcast = np.broadcast_to(c, Y.shape)

    # Build lexicographic key (negative for descending order)
    dtype = np.dtype([("y", Y.dtype), ("c", c.dtype)])
    keys = np.empty(Y.shape, dtype=dtype)
    keys["y"] = -Y
    keys["c"] = -c_broadcast

    # Step 1: select top-k candidates per row (unordered but correct set)
    topk_unsorted = np.argpartition(keys, k - 1, axis=1)[:, :k]

    # Step 2: extract their keys for final ordering
    topk_keys = np.take_along_axis(keys, topk_unsorted, axis=1)

    # Step 3: sort within top-k using full lexicographic rule
    order = np.argsort(topk_keys, axis=1)

    # Step 4: apply ordering
    topk_sorted = np.take_along_axis(topk_unsorted, order, axis=1)

    return topk_sorted


def agreement_with_bias_only(topk_recs, bias_only):
    """
    On average, the share of the same recommended items for a given user as the bias-only model.
    1 = same recommendations as bias-only model (up to ordering)
    0 = each user is recommended completely different books by both models
    """
    _, K = topk_recs.shape
    row_overlaps = (
        (bias_only[:, :, None] == topk_recs[:, None, :]).any(axis=2).sum(axis=1)
    )
    return row_overlaps.mean() / K


def mean_overlap(topk_recs):
    """
    Expected share of shared items between two random users.
    """
    n, K = topk_recs.shape
    counts = np.bincount(topk_recs.ravel())
    mean_overlap = (counts * (counts - 1)).sum() / (n * (n - 1))
    return mean_overlap / K


def coverage(topk_recs, m):
    """
    Total number of distinct recommended books. Maximum = n*K / m.
    """
    return len(np.unique(topk_recs)) / m


def rel_log_arp(topk_recs, log_pop, base):
    """
    How popular (in terms of number of ratings) the recommended items are, on average.
    Log version - prevents a few blockbuster books from dominating the metric.
    > 0 - model favors more popular books than base
    < 0 - model favors less popular books than base

    base = "uniform": Am I recommending more popular items than a random item
        from the catalog?

    base = "data": Am I recommending more popular items than an item drawn
        proportionally to how often users interact with it?
    """
    model_score = log_pop[topk_recs].mean()  # shape (n_users, K)
    return model_score - base


def rel_mean_rating(topk_recs, mean_ratings, global_mean_rating):
    """
    > 1 - on average, books with higher than average ratings are recommended
    < 1 - on average, books with lower than average ratings are recommended
    """
    return mean_ratings[topk_recs].mean() / global_mean_rating


def entropy(topk_recs, m):
    """
    How evenly the recommendations are distributed across items.
    Low entropy - few items dominate, strong popularity bias, low diversity
    High entropy - recommendations spread across many items, more diverse exposure
    """
    n, K = topk_recs.shape

    counts = np.bincount(topk_recs.ravel())
    counts = counts[counts > 0]

    p = counts / (n * K)

    return (-np.sum(p * np.log(p))) / np.log(m)


def update_best_sofar(
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
):

    best_sofar["mu"] = mu.copy()
    best_sofar["b"] = b.copy()
    best_sofar["c"] = c.copy()
    best_sofar["U"] = U.copy()
    best_sofar["V"] = V.copy()
    best_sofar["valid_rmse"] = valid_rmse
    best_sofar["valid_ndcg"] = valid_ndcg
    best_sofar["counter"] = counter
    best_sofar["max_norm_diff"] = max_norm_diff
    best_sofar["valid_rmse_diff"] = valid_rmse_diff
    best_sofar["valid_ndcg_diff"] = valid_ndcg_diff
    return best_sofar
