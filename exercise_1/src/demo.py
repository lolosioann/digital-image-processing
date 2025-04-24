import numpy as np
from PIL import Image

from hist_modif import perform_hist_eq, perform_hist_matching
from hist_utils import (
    calculate_hist_of_img,
    show_histogram,
)

# flags for testing - will be removed later
TEST_HIST_EQ = True
TEST_HIST_MATCHING = False
modes = ["greedy", "non-greedy", "post-disturbance"]


# load images and convert to grayscale in [0, 1]
input_img_path = "/home/johnlolos/Coding/dip/exercise_1/src/input_img.jpg"
ref_img_path = "/home/johnlolos/Coding/dip/exercise_1/src/ref_img.jpg"
input_img = Image.open(input_img_path).convert("L")
ref_img = Image.open(ref_img_path).convert("L")
input_img = np.array(input_img).astype(np.float64) / 255
ref_img = np.array(ref_img).astype(np.float64) / 255

# perform hist equalization for each mode
# and show results
if TEST_HIST_EQ:
    for mode in modes:
        eq_img = perform_hist_eq(input_img, mode=mode)
        # show the image and its histogram
        eq_hist = calculate_hist_of_img(eq_img, return_normalized=True)
        show_histogram(eq_hist, title=f"Histogram Equalization ({mode})")
        eq_img_pil = Image.fromarray((eq_img * 255).astype(np.uint8))
        eq_img_pil.show(title=f"Histogram Equalization ({mode})")


# perform hist matching for each mode
# and show results
if TEST_HIST_MATCHING:
    for mode in modes:
        matched_img = perform_hist_matching(input_img, ref_img, mode=mode)
        # show the image and its histogram
        matched_hist = calculate_hist_of_img(
            matched_img, return_normalized=True
        )
        show_histogram(matched_hist, title=f"Histogram Matching ({mode})")
        matched_img_pil = Image.fromarray((matched_img * 255).astype(np.uint8))
        matched_img_pil.show(title=f"Histogram Matching ({mode})")
