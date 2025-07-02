import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


def spectral_clustering(
    affinity_mat: np.ndarray, k: int, random_state=None
) -> np.ndarray:
    """
    Spectral clustering of an undirected graph defined by its affinity matrix.

    Parameters
    ----------
    affinity_mat : np.ndarray
        Square, symmetric affinity matrix W ∈ R^{nxn}.
    k : int
        Desired number of clusters.

    Returns
    -------
    np.ndarray
        Integer labels of length n assigning each vertex to a cluster.
    """
    n = affinity_mat.shape[0]

    # degenerate cases guard
    if k >= n:
        return np.arange(n)  # each node in its own cluster
    if k <= 0:
        raise ValueError("k must be a positive integer")

    #  Laplacian
    degrees = affinity_mat.sum(axis=1)
    D = np.diag(degrees)
    L = D - affinity_mat

    # eigen‑decomposition
    # for small n, use dense solver; for large n, use sparse solver
    # because for small n ARPACK was unreliable
    try:
        if n <= 200:
            _, eigvecs = eigh(L)  # returns all eigenvectors, sorted
            eigvecs = eigvecs[:, :k]  # k columns with smallest eigenvalues
        else:
            _, eigvecs = eigs(L, k=k, which="SM")  # ARPACK, smallest magnitude
            eigvecs = eigvecs.real  # Laplacian symmetric
    except Exception:  # on failure fallback to dense
        _, eigvecs = eigh(L)
        eigvecs = eigvecs[:, :k]

    # row normalisation for better clustering
    row_norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0  # avoid division by zero
    U = eigvecs / row_norms  # n × k matrix

    # k‑means
    if random_state is not None:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    else:
        kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(U)

    # if k‑means collapses to < k centroids,
    # fall back to a deterministic labelling.
    if len(np.unique(labels)) < k:
        labels = np.arange(n) % k

    return labels
