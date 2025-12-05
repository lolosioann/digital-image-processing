from scipy.io import loadmat

from src.part3.spectral_clustering import spectral_clustering
from src.utils.paths import data_path_str
import os


def demo_path(p: str) -> str:
    """
    Resolve a demo-local relative path to an absolute repo path.
    Use like: imread(demo_path("images/my_image.png"))
    """
    if not isinstance(p, str):
        return p
    if os.path.isabs(p):
        return p
    pkg = (__package__.split(".")[-1]) if __package__ else "part3"
    return data_path_str("src", pkg, *p.split("/")).__str__()


data = loadmat(demo_path("dip_hw_3.mat"))
d1a = data["d1a"]
random_state = 1

for i in range(2, 5):
    print(f"Running spectral clustering with k={i}...")
    labels = spectral_clustering(d1a, i, random_state=random_state)
    print(f"Labels for k={i}: {labels}")
    print(f"Number of unique labels: {len(set(labels))}\n")
