# @author : ThinkPad 
# @date : 2023/2/20

import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch
import sys
from tqdm import tqdm
from typing import Union


def _h_beta_torch(D: torch.Tensor, beta: float = 1.0):
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP + 1e-12) + beta * torch.sum(D * P) / (sumP + 1e-12)
    P = P / (sumP + 1e-12)
    return H, P


# noinspection PyPep8Naming
def _x2p_torch(X: torch.Tensor, tol: float = 1e-5, perplexity: float = 30.0,
               device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               verbose: bool = False):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    # print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X * X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n, device=device)
    beta = torch.ones(n, 1, device=device)
    logU = torch.log(torch.tensor([perplexity], device=device))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    bar = range(n)
    if verbose:
        bar = tqdm(bar)

    for i in bar:
        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i] + n_list[i + 1:n]]

        (H, thisP) = _h_beta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        h_diff = H - logU
        tries = 0
        while torch.abs(h_diff) > tol and tries < 50:
            # If not, increase or decrease precision
            if h_diff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = _h_beta_torch(Di, beta[i])

            h_diff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i] + n_list[i + 1:n]] = thisP

    # Return final P-matrix
    return P


def _pca_torch(X, no_dims=50):
    # print("Preprocessing the data using PCA...")
    # (n, d) = X.shape
    X = X - torch.mean(X, 0)

    # (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # # split M real
    # for i in range(d):
    #     if l[i, 1] != 0:
    #         M[:, i + 1] = M[:, i]
    #         i += 1
    #
    # Y = torch.mm(X, M[:, 0:no_dims])
    (l, M) = torch.linalg.eigh(torch.mm(X.t(), X))
    Y = torch.mm(X, M[:, -no_dims:])
    return Y


# noinspection PyPep8Naming
def tsne(X: Union[torch.Tensor, np.ndarray], no_dims: int = 2, initial_dims: int = 50, perplexity: float = 30.0,
         max_iter: int = 1000, verbose: bool = False,
         device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    if not isinstance(no_dims, int) or no_dims <= 0:
        raise ValueError("dims must be positive integer")
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"using {device}", file=sys.stderr)

    X = X.to(device)

    if verbose:
        print("initializing...", file=sys.stderr)
    # Initialize variables
    if initial_dims < X.shape[1]:
        X = _pca_torch(X, initial_dims)
    elif verbose:
        print("skipping PCA because initial_dims is larger than input dimensionality", file=sys.stderr)
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims, device=device)
    iY = torch.zeros(n, no_dims, device=device)
    gains = torch.ones(n, no_dims, device=device)

    # Compute P-values
    if verbose:
        print("computing p-values...", file=sys.stderr)
    P = _x2p_torch(X, 1e-5, perplexity, device=device, verbose=verbose)
    P = P + P.t()
    P = P / (torch.sum(P) + 1e-12)
    P = P * 4.  # early exaggeration
    P = torch.max(P, torch.tensor(1e-21, device=P.device))  # (N, N)

    if verbose:
        print("fitting...", file=sys.stderr)

    bar = range(max_iter)

    if verbose:
        bar = tqdm(bar, desc='fitting tsne')

    for it in bar:
        # Compute pairwise affinities
        sum_Y = torch.sum(Y * Y, dim=1)
        num = -2. * torch.mm(Y, Y.t())  # (N, N)
        num = 1. / (1. + (num + sum_Y).t() + sum_Y)
        num.fill_diagonal_(0)
        Q = num / (torch.sum(num) + 1e-12)
        Q = torch.max(Q, torch.tensor(1e-12, device=Q.device))

        # Compute gradient
        PQ = P - Q
        # ((N, 1, N).repeat -> (N, no_dims, N).permute -> (N, N, no_dims)) *
        # ((N, no_dims) - (N, 1, no_dims) -> (N, N, no_dims))
        # -> (N, N, no_dims).sum[1] -> (N, no_dims)

        dY = torch.sum(
            (PQ * num).unsqueeze(1).repeat(1, no_dims, 1).transpose(2, 1) * (Y.unsqueeze(1) - Y),
            dim=1
        )

        # Perform the update
        if it < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).float() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).float()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if verbose:
            C = torch.sum(P * torch.log(P / Q + 1e-12))
            bar.set_description(f"fitting tsne .... error: {C.cpu().item():.3f}")

        # Stop lying about P-values
        if it == 100:
            P = P / 4.

    # Return solution
    return Y
