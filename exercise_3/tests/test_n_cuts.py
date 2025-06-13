import numpy as np
import pytest

from n_cuts import (  # adjust the import path if needed
    calculate_n_cut_value,
    n_cuts,
)


def test_two_connected_components():
    """
    Test on a graph with two disconnected clusters.
    The algorithm should find exactly two clusters.
    """
    W = np.zeros((6, 6))
    W[0, 1] = W[1, 0] = 1
    W[1, 2] = W[2, 1] = 1
    W[3, 4] = W[4, 3] = 1
    W[4, 5] = W[5, 4] = 1

    labels = n_cuts(W, k=2, random_state=42)
    assert len(np.unique(labels)) == 2


def test_complete_graph():
    """
    All nodes are fully connected. Spectral clustering will depend on symmetry breaking,
    but labels must still contain exactly k distinct classes.
    """
    W = np.ones((10, 10)) - np.eye(10)  # fully connected
    k = 3
    labels = n_cuts(W, k, random_state=42)
    assert len(np.unique(labels)) == k


def test_k_equals_n():
    """
    When k == n, each node should be in its own cluster (fallback).
    """
    W = np.eye(5)
    labels = n_cuts(W, k=5)
    assert np.array_equal(labels, np.arange(5))


def test_k_greater_than_n():
    """
    When k > n, fallback labeling should apply: one cluster per node.
    """
    W = np.eye(4)
    labels = n_cuts(W, k=6)
    assert np.array_equal(labels, np.arange(4))


def test_k_is_zero():
    """
    When k == 0, function should raise ValueError.
    """
    W = np.eye(3)
    with pytest.raises(ValueError):
        n_cuts(W, 0)


def test_empty_affinity():
    """
    A graph with all-zero affinity matrix should return mod-n labeling.
    """
    W = np.zeros((6, 6))
    labels = n_cuts(W, 3)
    expected = np.arange(6) % 3
    assert np.array_equal(labels, expected)


def test_reproducibility():
    """
    Random_state should produce reproducible labelings.
    """
    rng = 42
    W = np.random.rand(20, 20)
    W = 0.5 * (W + W.T)  # make symmetric
    np.fill_diagonal(W, 0)

    labels1 = n_cuts(W, 4, random_state=rng)
    labels2 = n_cuts(W, 4, random_state=rng)
    assert np.array_equal(labels1, labels2)


def test_cluster_assignment_shape():
    """
    Output labels should match input size.
    """
    W = np.random.rand(12, 12)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    labels = n_cuts(W, 3)
    assert labels.shape == (12,)


def test_simple_case():
    # Απλό παράδειγμα 2 ομάδων με 4 κόμβους
    W = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
    labels = np.array([1, 1, 0, 0])  # 2 ομάδες
    ncut_value = calculate_n_cut_value(W, labels)
    expected_ncut = 2.0 - (2.0 / 2.0 + 2.0 / 2.0)  # Παράδειγμα υπολογισμού
    assert np.isclose(ncut_value, expected_ncut, atol=1e-4)


def test_no_connection():
    # Χωρίς συνδέσεις μεταξύ κόμβων (η Ncut πρέπει να είναι 2)
    W = np.zeros((4, 4))  # Χωρίς βαρύτητα
    labels = np.array([0, 0, 1, 1])  # 2 ομάδες
    ncut_value = calculate_n_cut_value(W, labels)
    assert np.isclose(ncut_value, 2.0)


def test_complete_graph():
    # Πλήρης γραφή με συνδέσεις μεταξύ όλων των κόμβων
    W = np.ones((4, 4)) - np.eye(4)  # Πλήρης γραφή χωρίς τις διαγώνιες
    labels = np.array([0, 0, 1, 1])  # 2 ομάδες
    ncut_value = calculate_n_cut_value(W, labels)
    expected_ncut = 2.0 - (4.0 / 4.0 + 4.0 / 4.0)
    assert np.isclose(ncut_value, expected_ncut, atol=1e-4)


def test_single_group():
    # Όλοι οι κόμβοι ανήκουν στην ίδια ομάδα
    W = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
    labels = np.array([1, 1, 1, 1])  # Όλοι στην ίδια ομάδα
    ncut_value = calculate_n_cut_value(W, labels)
    assert np.isclose(ncut_value, 0.0)


def test_edge_case_empty_graph():
    # Περίπτωση κενής γραφής (χωρίς κόμβους)
    W = np.array([[]])
    labels = np.array([])
    ncut_value = calculate_n_cut_value(W, labels)
    assert np.isclose(ncut_value, 2.0)


def test_single_edge_case():
    # Μόνο 2 κόμβοι συνδεδεμένοι
    W = np.array([[0, 1], [1, 0]])
    labels = np.array([0, 1])  # Δύο ομάδες
    ncut_value = calculate_n_cut_value(W, labels)
    expected_ncut = 2.0 - (
        0.0 / 1.0 + 0.0 / 1.0
    )  # Συνδέσεις μόνο μεταξύ των ομάδων
    assert np.isclose(ncut_value, expected_ncut, atol=1e-4)


# Δοκιμές για σφάλματα
def test_invalid_affinity_matrix():
    # Εσφαλμένη διάσταση affinity matrix
    W = np.array([[0, 1, 1], [1, 0, 1]])  # Δεν είναι τετράγωνος πίνακας
    labels = np.array([0, 1, 0])
    with pytest.raises(ValueError):
        calculate_n_cut_value(W, labels)


def test_invalid_labels():
    # Εσφαλμένες ετικέτες (πρέπει να είναι 0 ή 1)
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    labels = np.array([0, 2, 0])  # Ετικέτα "2" είναι μη έγκυρη
    with pytest.raises(ValueError):
        calculate_n_cut_value(W, labels)


def test_mismatched_affinity_and_labels():
    # Μη συμβατό μέγεθος affinity matrix και labels
    W = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    labels = np.array([0, 1])  # Λείπει μια ετικέτα
    with pytest.raises(ValueError):
        calculate_n_cut_value(W, labels)


if __name__ == "__main__":
    pytest.main()
