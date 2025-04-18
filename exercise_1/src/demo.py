import numpy as np
from PIL import Image

from hist_modif import perform_hist_modification
from hist_utils import *

input_img_path = "/home/johnlolos/Coding/dip/exercise_1/src/input_img.jpg"
ref_img_path = "/home/johnlolos/Coding/dip/exercise_1/src/ref_img.jpg"

input_img = Image.open(input_img_path).convert("L")
ref_img = Image.open(ref_img_path).convert("L")
input_img = np.array(input_img).astype(np.float64) / 255
ref_img = np.array(ref_img).astype(np.float64) / 255

input_hist = calculate_hist_of_img(input_img, return_normalized=True)
ref_hist = calculate_hist_of_img(ref_img, return_normalized=True)
show_histogram(input_hist, "Input Image Histogram")
show_histogram(ref_hist, "Reference Image Histogram")


out = perform_hist_modification(input_img, ref_hist, mode="post-disturbance")
out_hist = calculate_hist_of_img(out, return_normalized=True)
show_histogram(out_hist, "Output Image Histogram")
out = (out * 255).astype(np.uint8)
out = Image.fromarray(out)
out.show()

# bring images back to [0, 255] range
input_img = (input_img * 255).astype(np.uint8)
ref_img = (ref_img * 255).astype(np.uint8)

# display images
input_img = Image.fromarray(input_img.astype(np.uint8))
ref_img = Image.fromarray(ref_img.astype(np.uint8))
# input_img.show()
# ref_img.show()
