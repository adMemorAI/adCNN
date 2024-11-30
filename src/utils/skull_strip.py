# utils/config.py

import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_erosion, disk
from scipy.ndimage import binary_fill_holes

class SkullStrip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        # img is a PIL Image
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        # Convert PIL Image to numpy array
        gray_image = np.array(img)

        # Apply Gaussian smoothing
        smoothed_image = gaussian(gray_image, sigma=2)

        # Edge detection using canny edge detector
        edges = canny(smoothed_image, sigma=1)

        # Apply binary closing
        closed_edges = binary_closing(edges, disk(5))

        # Fill holes
        filled_mask = binary_fill_holes(closed_edges)

        # Erosion
        eroded_mask = binary_erosion(filled_mask, disk(8))

        # Create outer boundary mask
        outer_boundary_mask = filled_mask ^ eroded_mask

        # Invert the mask
        final_mask = np.logical_not(outer_boundary_mask)

        # Convert mask to uint8
        final_mask_uint8 = final_mask.astype(np.uint8)

        # Multiply the mask with the grayscale image
        final_image = gray_image * final_mask_uint8

        # Convert back to PIL Image
        final_image = Image.fromarray(final_image.astype(np.uint8))

        return final_image

