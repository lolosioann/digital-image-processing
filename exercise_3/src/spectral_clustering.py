import numpy as np
from scipy.linalg import eigh  # dense, symmetric
from scipy.sparse.linalg import eigs  # iterative (ARPACK)
from sklearn.cluster import KMeans


def spectral_clustering(
    affinity_mat: np.ndarray, k: int, random_state=None
) -> np.ndarray:
    """
    Spectral clustering of an undirected graph defined by its affinity matrix.

    Parameters
    ----------
    affinity_mat : np.ndarray
        Square, symmetric affinity matrix W ∈ ℝ^{n×n}.
    k : int
        Desired number of clusters.

    Returns
    -------
    np.ndarray
        Integer labels of length n assigning each vertex to a cluster.
    """
    n = affinity_mat.shape[0]

    # ────────────── trivial / degenerate cases ──────────────
    if k >= n:  # e.g. identity affinity, each vertex isolated
        return np.arange(n)  # each node in its own cluster
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # ────────────── Laplacian construction ──────────────
    degrees = affinity_mat.sum(axis=1)
    D = np.diag(degrees)
    L = D - affinity_mat  # unnormalised Laplacian (specification §2)

    # ────────────── eigen‑decomposition ──────────────
    # For small dense matrices a full symmetric solver is both faster and
    # numerically safer than ARPACK.
    try:
        if n <= 200:
            _, eigvecs = eigh(L)  # returns all eigenvectors, sorted
            eigvecs = eigvecs[:, :k]  # k columns with smallest eigenvalues
        else:
            _, eigvecs = eigs(L, k=k, which="SM")  # ARPACK, smallest magnitude
            eigvecs = eigvecs.real  # Laplacian is real symmetric
    except Exception:  # any failure → dense fallback
        _, eigvecs = eigh(L)
        eigvecs = eigvecs[:, :k]

    # ────────────── row normalisation ──────────────
    row_norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0  # avoid division by zero
    U = eigvecs / row_norms  # n × k matrix

    # ────────────── k‑means in the spectral space ──────────────
    if random_state is not None:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    else:
        kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(U)

    # Rare corner‑case: if k‑means collapses to < k centroids (can happen for
    # degenerate spectra), fall back to a deterministic labelling.
    if len(np.unique(labels)) < k:
        labels = np.arange(n) % k

    return labels
