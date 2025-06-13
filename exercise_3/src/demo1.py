from scipy.io import loadmat

from spectral_clustering import spectral_clustering

data = loadmat("dip_hw_3.mat")
d1a = data["d1a"]
random_state = 1

for i in range(2, 5):
    print(f"Running spectral clustering with k={i}...")
    labels = spectral_clustering(d1a, i, random_state=random_state)
    print(f"Labels for k={i}: {labels}")
    print(f"Number of unique labels: {len(set(labels))}\n")
