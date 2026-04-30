import numpy as np


def initialize_mu_b_c(Y):
    O_mask = Y > 0
    Y_masked = Y * O_mask

    mu = Y[O_mask].mean()

    sums1 = np.sum(Y_masked, axis=1)
    counts1 = np.count_nonzero(Y_masked, axis=1)
    no_ratings1 = counts1 == 0
    counts1[no_ratings1] = 1
    b = sums1 / counts1 - mu
    b[no_ratings1] = 0  # not -mu

    sums0 = np.sum(Y_masked, axis=0)
    counts0 = np.count_nonzero(Y_masked, axis=0)
    no_ratings0 = counts0 == 0
    counts0[no_ratings0] = 1
    c = sums0 / counts0 - mu
    c[no_ratings0] = 0  # not -mu

    return mu, b, c
