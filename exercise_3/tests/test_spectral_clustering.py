import numpy as np
import pytest
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances

# Import the spectral_clustering function
from spectral_clustering import (
    spectral_clustering,  # Adjust import according to your project structure
)


# 1. Basic Test on a Simple Synthetic Affinity Matrix
def test_basic_clustering():
    # Create a small synthetic affinity matrix for a 4-node graph
    affinity_mat = np.array(
        [
            [1, 0.8, 0.5, 0.3],
            [0.8, 1, 0.4, 0.2],
            [0.5, 0.4, 1, 0.7],
            [0.3, 0.2, 0.7, 1],
        ]
    )

    k = 2  # Let's cluster into 2 groups
    labels = spectral_clustering(affinity_mat, k)

    assert labels.shape[0] == 4  # We should have 4 nodes (rows) in the output
    assert len(np.unique(labels)) == 2  # Expecting 2 clusters

    # The expected labels could depend on the graph structure,
    # so we can print labels for manual inspection if needed
    print("Cluster labels:", labels)


# 2. Test on Random Affinity Matrix (Random Graph)
def test_random_affinity_matrix():
    # Random 10x10 matrix with values between 0 and 1, representing affinity
    np.random.seed(42)
    affinity_mat = np.random.rand(10, 10)
    affinity_mat = (affinity_mat + affinity_mat.T) / 2  # Make it symmetric
    np.fill_diagonal(affinity_mat, 1)  # Self-loop with affinity 1

    k = 3  # Let's cluster into 3 groups
    labels = spectral_clustering(affinity_mat, k)

    assert (
        labels.shape[0] == 10
    )  # We should have 10 nodes (rows) in the output
    assert len(np.unique(labels)) == 3  # Expecting 3 clusters

    print("Cluster labels:", labels)


# 3. Test on Data with Known Structure (using synthetic 2D data like circles)
def test_synthetic_circles():
    # Generate synthetic data with two clusters in a 2D plane
    X, _ = make_circles(n_samples=100, factor=0.5, noise=0.1)

    # Compute the pairwise distance matrix as the affinity matrix
    affinity_mat = np.exp(
        -(pairwise_distances(X) ** 2) / (2.0 * 0.1**2)
    )  # Using Gaussian kernel

    k = 2  # We know the data has 2 clusters
    labels = spectral_clustering(affinity_mat, k)

    assert labels.shape[0] == 100  # We should have 100 points (rows)
    assert len(np.unique(labels)) == 2  # Expecting 2 clusters

    # Print cluster labels for manual inspection
    print("Cluster labels:", labels)


# 4. Edge Case: Identity Matrix (all pixels are equally similar)
def test_identity_matrix():
    # Create a simple identity matrix (all nodes are isolated)
    affinity_mat = np.identity(5)  # 5 nodes, no affinity between them

    k = 1  # Only 1 cluster expected (trivial case)
    labels = spectral_clustering(affinity_mat, k)

    assert labels.shape[0] == 5  # 5 nodes
    assert len(np.unique(labels)) == 1  # Expecting only 1 cluster

    print("Cluster labels:", labels)


# 5. Test on a Disconnected Graph (all zeros except the diagonal)
def test_disconnected_graph():
    # Create a disconnected graph with affinity = 0 except diagonal (self-loops)
    affinity_mat = np.identity(6)

    k = 6  # Each node should be its own cluster
    labels = spectral_clustering(affinity_mat, k)

    assert labels.shape[0] == 6  # 6 nodes
    assert (
        len(np.unique(labels)) == 6
    )  # Expecting 6 clusters (each node isolated)

    print("Cluster labels:", labels)
