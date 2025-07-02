import numpy as np
from scipy.linalg import LinAlgError, eigh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


def n_cuts(affinity_mat: np.ndarray, k: int, random_state=None) -> np.ndarray:
    """
    Spectral clustering via the N-cuts criterion.

    Parameters
    ----------
    affinity_mat : np.ndarray
        Square, symmetric affinity matrix W ∈ R^{nxn}.
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

    # trivial cases guard
    if k >= n:
        return np.arange(n)
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Laplacian
    degrees = affinity_mat.sum(axis=1)
    D = np.diag(degrees)
    L = D - affinity_mat

    # generalised eigen‑problem  Lx = lambda*D*x
    dense_threshold = 200  # switch to dense solver below this
    try:
        if n <= dense_threshold:
            _, eigvecs = eigh(L, D, subset_by_index=(0, k - 1))
        else:
            L_sp = csr_matrix(L)
            D_sp = diags(degrees)
            _, eigvecs = eigsh(
                L_sp, k=k, M=D_sp, which="SM"
            )  # keep k with smallest magnitude
    except (LinAlgError, Exception) as e:
        # fallback: dense generalised solver
        print(f"Eigenvalue computation failed, falling back: {e}")
        _, eigvecs = eigh(L, D, subset_by_index=(0, k - 1))

    # row normalisation for better clustering
    row_norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    U = eigvecs / row_norms

    # k‑means
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state if random_state is not None else 1,
    )
    labels = km.fit_predict(U)

    # k‑means collapsed → force unique labelling
    if len(np.unique(labels)) < k:
        print("Warning: k-means collapsed, forcing unique labels.")
        labels = np.arange(n) % k

    return labels


def calculate_n_cut_value(
    affinity_mat: np.ndarray, cluster_idx: np.ndarray
) -> float:
    """
    Compute the Ncut value for a 2-way partition of a graph.

    Parameters
    ----------
    affinity_mat : np.ndarray
        Symmetric affinity matrix (n x n) of the graph.
    cluster_idx : np.ndarray
        Binary array of shape (n,), where each entry is
        0 or 1 indicating
        to which of the two clusters a node belongs.

    Returns
    -------
    float
        The Ncut value of the partition. Lower values
        indicate better clustering.
        Returns NaN if the cut is degenerate or invalid.
    """

    # Check validity of inputs
    if affinity_mat.shape[0] != affinity_mat.shape[1]:
        raise ValueError(
            f"Affinity matrix must be square. Got shape {affinity_mat.shape}."
        )
    if affinity_mat.shape[0] != cluster_idx.size:
        raise ValueError(
            "Affinity matrix size does not match the number of labels."
        )
    if affinity_mat.size == 0:
        return 2.0  # Maximum possible Ncut for an empty graph

    # Create binary masks for the two clusters A and B
    A = cluster_idx == 1
    B = cluster_idx == 0
    V = np.ones_like(cluster_idx, dtype=bool)  # full graph mask

    # Compute associations:
    # assoc(X, Y) = sum of edge weights between nodes in X and Y
    assoc_AA = np.sum(affinity_mat[A][:, A])
    assoc_BB = np.sum(affinity_mat[B][:, B])
    assoc_AV = np.sum(affinity_mat[A][:, V])
    assoc_BV = np.sum(affinity_mat[B][:, V])

    # to be safe
    if assoc_AV == 0 or assoc_BV == 0:
        return np.nan

    # Compute Nassoc, then Ncut = 2 - Nassoc
    Nassoc = assoc_AA / assoc_AV + assoc_BB / assoc_BV
    return 2.0 - Nassoc


def n_cuts_recursive(
    affinity_mat: np.ndarray, T1: int, T2: float, random_state=None
) -> np.ndarray:
    """
    Recursively partition a graph using 2-way Normalized Cuts until
    stopping criteria are met.

    Parameters
    ----------
    affinity_mat : np.ndarray
        Symmetric affinity matrix (n x n) representing the graph.
    T1 : int
        Minimum number of nodes required in a cluster to allow
        further partitioning.
    T2 : float
        Maximum acceptable Ncut value for a split to be valid.
    random_state : int, optional
        Seed used for k-means clustering to ensure reproducibility.

    Returns
    -------
    np.ndarray
        An array of final cluster labels for each node in the graph.
    """

    n = affinity_mat.shape[0]
    labels = -np.ones(n, dtype=int)  # Final labels to be filled in
    next_label = [0]  # Mutable label index counter

    def recursive_split(indices: np.ndarray):
        """
        Inner function to recursively split a subgraph defined by
        the given node indices.
        """

        # Extract sub-affinity matrix corresponding to the current subset
        W = affinity_mat[np.ix_(indices, indices)]

        # Perform 2-way normalized cut on the subgraph
        cluster = n_cuts(W, k=2, random_state=random_state)

        # If the result is degenerate (i.e., one group only),
        # assign all to the same label
        if len(np.unique(cluster)) < 2:
            labels[indices] = next_label[0]
            next_label[0] += 1
            return

        # Map subgraph clustering results back to full graph index space
        group0 = indices[cluster == 0]
        group1 = indices[cluster == 1]

        # Compute the Ncut value of the proposed partition
        ncut = calculate_n_cut_value(W, cluster)

        # Logging diagnostic info about this split
        print("Splitting indices")
        print(
            f"  → Group0 size: {len(group0)}, "
            f"Group1 size: {len(group1)}, "
            f"Ncut: {ncut:.4f}"
        )

        # Apply stopping criteria:
        if len(group0) < T1 or len(group1) < T1 or ncut > T2:
            # Stop recursion; assign current group to a unique label
            labels[indices] = next_label[0]
            next_label[0] += 1
            return

        # Recursively split each subgroup
        recursive_split(group0)
        recursive_split(group1)

    # Start recursive splitting from the full set of nodes
    recursive_split(np.arange(n))
    return labels
