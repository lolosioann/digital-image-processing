import numpy as np
from scipy.linalg import LinAlgError, eigh
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
    dense_threshold = 200  # heuristic: switch to dense solver below this
    try:
        if n <= dense_threshold:  # small → dense
            # eigh returns eigenvalues ASC; we keep the k smallest
            _, eigvecs = eigh(L, D, subset_by_index=(0, k - 1))
        else:  # large → sparse
            # ARPACK: symmetric solver with mass matrix M = D
            L_sp = csr_matrix(L)
            D_sp = diags(degrees)
            _, eigvecs = eigsh(
                L_sp, k=k, M=D_sp, which="SM"
            )  # smallest magnitude
    except (LinAlgError, Exception) as e:
        # fallback: dense generalised solver
        print(f"Eigenvalue computation failed, falling back: {e}")
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
        print("Warning: k-means collapsed, forcing unique labels.")
        labels = np.arange(n) % k

    return labels


def calculate_n_cut_value(
    affinity_mat: np.ndarray, cluster_idx: np.ndarray
) -> float:
    # Έλεγχος για το αν ο affinity πίνακας είναι τετράγωνος
    if affinity_mat.shape[0] != affinity_mat.shape[1]:
        raise ValueError(
            f"Affinity matrix must be square. Got shape {affinity_mat.shape}."
        )

    if affinity_mat.shape[0] != cluster_idx.size:
        raise ValueError(
            "Affinity matrix size does not match the number of labels."
        )

    if affinity_mat.size == 0:
        return 2  # Or another reasonable return value for an empty graph

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

    # Προστασία από διαιρέσεις με το μηδέν
    if assoc_AV == 0 or assoc_BV == 0:
        return np.nan  # Or another appropriate value

    # Υπολογισμός του Nassoc(A, B)
    Nassoc = assoc_AA / assoc_AV + assoc_BB / assoc_BV

    # Υπολογισμός του Ncut(A, B)
    Ncut = 2.0 - Nassoc

    return Ncut


# def n_cuts_recursive(affinity_mat: np.ndarray,
# T1: int, T2: float, random_state=None) -> np.ndarray:
#     n = affinity_mat.shape[0]
#     labels = -np.ones(n, dtype=int)
#     next_label = 0

#     stack = [np.arange(n)]

#     while stack:
#         indices = stack.pop()
#         W = affinity_mat[np.ix_(indices, indices)]

#         # Perform 2-way spectral clustering
#         cluster = n_cuts(W, k=2, random_state=random_state)

#         # Make sure it actually split the graph
#         unique = np.unique(cluster)
#         if len(unique) < 2:
#             labels[indices] = next_label
#             next_label += 1
#             continue

#         print(f"Splitting indices")
#         group0 = indices[cluster == 0]
#         group1 = indices[cluster == 1]

#         # Compute Ncut value of this split
#         ncut = calculate_n_cut_value(W, cluster)

#         print(f"  → Group0 size: {len(group0)},
# Group1 size: {len(group1)}, Ncut: {ncut:.4f}")
#         # Check stopping condition
#         if len(group0) < T1 or len(group1) < T1
# or ncut > T2 or np.isnan(ncut):
#             labels[indices] = next_label
#             next_label += 1
#         else:
#             stack.append(group0)
#             stack.append(group1)

#     return labels


def n_cuts_recursive(
    affinity_mat: np.ndarray, T1: int, T2: float, random_state=None
) -> np.ndarray:
    n = affinity_mat.shape[0]
    labels = -np.ones(n, dtype=int)
    next_label = [
        0
    ]  # Mutable label counter (so it updates across recursive calls)

    def recursive_split(indices: np.ndarray):
        W = affinity_mat[np.ix_(indices, indices)]

        cluster = n_cuts(W, k=2, random_state=random_state)

        unique = np.unique(cluster)
        if len(unique) < 2:
            labels[indices] = next_label[0]
            next_label[0] += 1
            return

        group0 = indices[cluster == 0]
        group1 = indices[cluster == 1]

        ncut = calculate_n_cut_value(W, cluster)

        print("Splitting indices")
        print(
            f"  → Group0 size: {len(group0)}, "
            f"Group1 size: {len(group1)}, "
            f"Ncut: {ncut:.4f}"
        )

        if len(group0) < T1 or len(group1) < T1 or ncut < T2 or np.isnan(ncut):
            labels[indices] = next_label[0]
            next_label[0] += 1
        else:
            recursive_split(group0)
            recursive_split(group1)

    # Start recursion on all indices
    recursive_split(np.arange(n))
    return labels
