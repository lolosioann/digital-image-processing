import numpy as np
from scipy.linalg import LinAlgError, eigh  # dense solver
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh  # symmetric ARPACK
from sklearn.cluster import KMeans


def n_cuts(affinity_mat: np.ndarray, k: int, random_state=None) -> np.ndarray:
    """
    Spectral clustering via the N‑cuts criterion.

    Parameters
    ----------
    affinity_mat : np.ndarray
        Square, symmetric affinity matrix W ∈ ℝ^{n×n}.
    k : int
        Desired number of clusters.
    random_state : int | None
        Seed forwarded to KMeans for reproducibility.

    Returns
    -------
    np.ndarray
        Integer labels of length n assigning each vertex to a cluster.
    """
    n = affinity_mat.shape[0]

    # ───── trivial / degenerate cases ─────
    if k >= n:
        return np.arange(n)
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # ───── Laplacian & degree matrix ─────
    degrees = affinity_mat.sum(axis=1)
    D = np.diag(degrees)
    L = D - affinity_mat  # unnormalised Laplacian

    # guard: fully disconnected graph ⇒ all degrees zero
    if np.allclose(degrees, 0):
        return np.arange(n) % k

    # ───── generalised eigen‑problem  Lx = λ D x ─────
    dense_threshold = 400  # heuristic: switch to dense solver below this
    try:
        if n <= dense_threshold:  # small → dense
            # eigh returns eigenvalues ASC; we keep the k smallest
            _, eigvecs = eigh(L, D, subset_by_index=(0, k - 1))
        else:  # large → sparse
            # ARPACK: symmetric solver with mass matrix M = D
            # M must be positive definite → degrees > 0 for fully connected graph
            L_sp = csr_matrix(L)
            D_sp = diags(degrees)
            _, eigvecs = eigsh(
                L_sp, k=k, M=D_sp, which="SM"
            )  # smallest magnitude
    except (LinAlgError, Exception):
        # fallback: dense generalised solver
        _, eigvecs = eigh(L, D, subset_by_index=(0, k - 1))

    # ───── row normalisation ─────
    row_norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    U = eigvecs / row_norms  # n × k

    # ───── k‑means in spectral space ─────
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state if random_state is not None else 1,
    )
    labels = km.fit_predict(U)

    # rare: k‑means collapsed → force unique labelling
    if len(np.unique(labels)) < k:
        labels = np.arange(n) % k

    return labels


def calculate_n_cut_value(
    affinity_mat: np.ndarray, cluster_idx: np.ndarray
) -> float:
    # Έλεγχος για το αν ο affinity πίνακας είναι τετράγωνος
    if affinity_mat.shape[0] != affinity_mat.shape[1]:
        raise ValueError("Affinity matrix must be square.")

    if affinity_mat.size == 0:
        return 2

    # Υπολογισμός του assoc(A, V), assoc(B, V) και assoc(A, A), assoc(B, B)
    A = cluster_idx == 1
    B = cluster_idx == 0
    V = np.ones_like(cluster_idx, dtype=bool)

    assoc_AA = np.sum(
        affinity_mat[A][:, A]
    )  # Στοιχεία που ανήκουν στην ομάδα A
    assoc_BB = np.sum(
        affinity_mat[B][:, B]
    )  # Στοιχεία που ανήκουν στην ομάδα B
    assoc_AV = np.sum(affinity_mat[A][:, V])  # Στοιχεία της A προς το σύνολο V
    assoc_BV = np.sum(affinity_mat[B][:, V])  # Στοιχεία της B προς το σύνολο V

    # Υπολογισμός του Nassoc(A, B)
    Nassoc = assoc_AA / assoc_AV + assoc_BB / assoc_BV

    # Υπολογισμός του Ncut(A, B)
    Ncut = 2.0 - Nassoc

    return Ncut
