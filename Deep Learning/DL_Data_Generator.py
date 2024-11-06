import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks


def process_image(image_path, idx, patch_size=64, visualize=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image {image_path}. Skipping.")
        return idx

    height, width = image.shape[:2]

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(image, (5, 5), 0).astype('uint8')

    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Apply Otsu's Thresholding
    ret, ot = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological Operations to Eliminate Noise
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh = cv2.erode(adaptive_thresh, kernel=kernel)
    adaptive_thresh = cv2.dilate(adaptive_thresh, kernel=kernel)

    # Create Gradient Mask to Isolate Specific Regions
    grad_x = cv2.Sobel(ot, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(ot, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_mask = (grad_magnitude == 0)
    dark_mask = (ot == 0)
    mask = gradient_mask & dark_mask

    # Apply Mask to Image
    img_copy = image.copy()
    img_copy[mask] = 0
    adaptive_thresh[mask] = 0

    # Skeletonization
    skeleton = skeletonize(adaptive_thresh > 0).astype(np.uint8) * 255

    # Hough Line Transform Parameters
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=True)
    h, theta, d = hough_line(skeleton, theta=tested_angles)

    # Extract Lines Using Hough Line Peaks
    lines = []
    threshold = 0.5 * h.max()
    num_peaks = 100  # Adjust based on expected number of grid lines
    accum, angles, dists = hough_line_peaks(h, theta, d, threshold=threshold, num_peaks=num_peaks)
    lines = list(zip(angles, dists))

    # Create a Blank Mask for Grid Lines
    grid_lines_mask = np.zeros_like(image, dtype=np.uint8)

    # Draw Detected Lines on the Grid Lines Mask
    for angle, dist in lines:
        # Convert polar coordinates to cartesian for line drawing
        a = np.cos(angle)
        b = np.sin(angle)
        x0 = a * dist
        y0 = b * dist
        # Define line length based on image size
        length = max(width, height)
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))
        cv2.line(grid_lines_mask, (x1, y1), (x2, y2), 255, 1)

    # Skeletonization on Grid Lines Mask
    grid_lines_skeleton = skeletonize(grid_lines_mask > 0).astype(np.uint8) * 255

    # Define Directories to Save Patches and Masks
    truth_patch_dir = "Deep Learning/Training/Skeleton"
    predict_patch_dir = "Deep Learning/Training/Raw"

    # Create directories if they don't exist
    os.makedirs(truth_patch_dir, exist_ok=True)
    os.makedirs(predict_patch_dir, exist_ok=True)

    # Segment the Image and Grid Lines Mask into Patches
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Define the patch boundaries
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)

            # Extract the image patch
            image_patch = image[y:y_end, x:x_end]

            # Extract the corresponding grid lines mask patch
            mask_patch = grid_lines_skeleton[y:y_end, x:x_end]

            # Check if the patch size is smaller than patch_size and pad if necessary
            if image_patch.shape[0] < patch_size or image_patch.shape[1] < patch_size:
                # Create a blank patch
                padded_image_patch = np.zeros((patch_size, patch_size), dtype=image_patch.dtype)
                padded_mask_patch = np.zeros((patch_size, patch_size), dtype=mask_patch.dtype)
                # Place the original patch in the top-left corner
                padded_image_patch[0:image_patch.shape[0], 0:image_patch.shape[1]] = image_patch
                padded_mask_patch[0:mask_patch.shape[0], 0:mask_patch.shape[1]] = mask_patch
                image_patch = padded_image_patch
                mask_patch = padded_mask_patch

            # Save the image patch
            image_patch_filename = os.path.join(
                predict_patch_dir, f"image_{idx}.png"
            )
            cv2.imwrite(image_patch_filename, image_patch)

            # Save the mask patch
            mask_patch_filename = os.path.join(
                truth_patch_dir,
                f"mask_{idx}.png"
            )
            cv2.imwrite(mask_patch_filename, mask_patch)

            # Visualization
            if visualize:
                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(image_patch, cmap='gray')
                axs[0].set_title("Image Patch")
                axs[0].axis('off')

                axs[1].imshow(mask_patch, cmap='gray')
                axs[1].set_title("Grid Lines Mask")
                axs[1].axis('off')

                plt.tight_layout()
                if visualize:
                    plt.show()

            idx += 1

    return idx


# Main Processing Loop with Visualization Options
def main():
    images_dir = "Deep Learning/TIFF"

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    idx = 0
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        print(f"Processing {image_file}...")

        # To visualize each patch, set visualize=True
        # To save visualizations, set save_visualizations=True
        # For large datasets, it's recommended to set visualize=False and save_visualizations=True
        idx = process_image(
            image_path,
            idx,
            patch_size=64,
            visualize=False
        )
        print(f"Finished processing {image_file}. Current index: {idx}")


if __name__ == "__main__":
    main()
