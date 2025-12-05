import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.part1.hist_modif import (
    perform_hist_eq,
    perform_hist_matching,
)
from src.part1.hist_utils import (
    calculate_hist_of_img,
    dict_to_hist_array,
    show_histogram,
)
from src.utils.paths import data_path_str


def demo_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    if os.path.isabs(p):
        return p
    pkg = (__package__.split(".")[-1]) if __package__ else "part1"
    return data_path_str("src", pkg, *p.split("/")).__str__()


def display_img_hist_grid(
    processed_img: np.ndarray,
    original_img: np.ndarray,
    processed_hist: Dict[float, float],
    original_hist: Dict[float, float],
    action: str,
    mode: str,
    save_image: bool = True,
    save_hist: bool = True,
) -> None:
    """
    Displays and optionally saves the result of histogram
    equalization or matching,
    including:
    - Reference image and its histogram
    - Processed image and its histogram
    In a 2x2 grid layout, this function generates and displays
    the images and their
    corresponding histograms, with an option to save them to disk.

    Args:
        processed_img (np.ndarray): Processed image (matched or equalized).
        original_img (np.ndarray): Original grayscale image.
        processed_hist (Dict[float, float]): Normalized histogram of the
            processed image.
        original_hist (Dict[float, float]): Normalized histogram
            of the original image.
        action (str): Action performed ("matching" or "equalization").
        mode (str): The mode used for histogram operation
            ('greedy', 'non-greedy', etc.).
        save_image (bool): If True, saves the processed image as a PNG file.
        save_hist (bool): If True, saves the histogram plot as a PNG file.
    """

    # Convert processed image to uint8 format for saving
    processed_img_uint8 = (processed_img * 255).astype(np.uint8)

    # Save the processed image to disk if specified
    if save_image:
        title = (
            f"matched_image_{mode}"
            if action == "matching"
            else f"equalized_image_{mode}"
        )
        Image.fromarray(processed_img_uint8).save(f"{title}.png")

    # Save the histogram plot of the processed image to disk if specified
    if save_hist:
        title = (
            f"matched_hist_{mode}" if action == "matching" else f"equalized_hist_{mode}"
        )
        show_histogram(processed_hist, title=title, save_path=f"{title}.png")

    # Convert the processed and original histograms to arrays for plotting
    processed_hist_arr = dict_to_hist_array(processed_hist)
    original_hist_arr = dict_to_hist_array(original_hist)

    # Create a 2x2 grid layout for displaying images and histograms
    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the original image
    axes[0, 0].imshow(original_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Plot the processed image
    axes[0, 1].imshow(processed_img, cmap="gray", vmin=0, vmax=1)
    title = (
        f"Matched Image ({mode})"
        if action == "matching"
        else f"Equalized Image ({mode})"
    )
    axes[0, 1].set_title(title)
    axes[0, 1].axis("off")

    # Plot the original histogram
    axes[1, 0].bar(
        np.linspace(0, 1, 256),
        original_hist_arr,
        width=1 / 255,
        color="green",
        alpha=0.7,
    )
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_xlabel("Gray Level")
    axes[1, 0].set_ylabel("Probability Density")

    # Plot the processed histogram
    axes[1, 1].bar(
        np.linspace(0, 1, 256),
        processed_hist_arr,
        width=1 / 255,
        color="blue",
        alpha=0.7,
    )
    title = (
        f"Matched Histogram ({mode})"
        if action == "matching"
        else f"Equalized Histogram ({mode})"
    )
    axes[1, 1].set_title(title)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xlabel("Gray Level")
    axes[1, 1].set_ylabel("Probability Density")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Load grayscale images and normalize pixel values to [0, 1]
input_img_path = demo_path("input_img.jpg")  # Path to the input image
ref_img_path = demo_path("ref_img.jpg")

input_img = Image.open(input_img_path).convert("L")  # Open and convert to grayscale
ref_img = Image.open(ref_img_path).convert("L")

# save ref_hist
ref_hist = calculate_hist_of_img(
    np.array(ref_img, dtype=np.float64) / 255.0, return_normalized=True
)  # Calculate histogram of reference image
# Save the histogram of the reference image
show_histogram(
    ref_hist,
    title="Reference Histogram",
    save_path="ref_hist.png",
)  # Save the histogram of the reference image

#  save the images
input_img.save("input_img.png")
ref_img.save("ref_img.png")

# Normalize images by dividing by 255 to scale pixel values to [0, 1]
input_img = np.array(input_img, dtype=np.float64) / 255.0
input_hist = calculate_hist_of_img(
    input_img, return_normalized=True
)  # Calculate histogram of input image

ref_img = np.array(ref_img, dtype=np.float64) / 255.0  # Normalize reference image

# Modes for histogram matching: greedy, non-greedy, and post-disturbance
MODES = ["greedy", "non-greedy", "post-disturbance"]

print("Press q to continue to the next image.")

# Apply histogram equalization for each mode and display/save results
for mode in MODES:
    eq_img = perform_hist_eq(input_img, mode=mode)  # Apply histogram equalization
    eq_hist = calculate_hist_of_img(
        eq_img, return_normalized=True
    )  # Calculate histogram for equalized image

    # Convert equalized image to uint8 format for saving
    eq_img_uint8 = (eq_img * 255).astype(np.uint8)

    # Save the histogram of the equalized image
    show_histogram(
        eq_hist,
        title=f"Histogram Equalization ({mode})",
        save_path=f"equalized_hist_{mode}.png",
    )

    # Display images and histograms for the equalized image
    display_img_hist_grid(
        eq_img,
        input_img,
        eq_hist,
        input_hist,
        mode=mode,
        action="equalization",
    )

# Apply histogram matching for each mode and display/save results
for mode in MODES:
    matched_img = perform_hist_matching(input_img, ref_img, mode=mode)
    matched_hist = calculate_hist_of_img(matched_img, return_normalized=True)
    ref_hist = calculate_hist_of_img(ref_img, return_normalized=True)

    # Display images and histograms for the matched image
    display_img_hist_grid(
        matched_img,
        ref_img,
        matched_hist,
        ref_hist,
        mode=mode,
        action="matching",
    )

print("Do you want to keep the saved images? (Y/n)")
save_images = input().strip().lower()

if save_images not in ["y", "n", ""]:
    print(save_images)
    print("Invalid input. Images will be kept by default.")

if save_images == "n":
    # Change to the script's directory
    script_dir = Path(__file__).resolve().parent
    if Path.cwd() != script_dir:
        os.chdir(script_dir)

    # Remove all .png files
    png_files = list(script_dir.glob("*.png"))
    if not png_files:
        print("No PNG files found to remove.")
    else:
        for file in png_files:
            try:
                file.unlink()
            except Exception as e:
                print(f"Failed to delete {file}: {e}")
        print("Saved images removed.")
