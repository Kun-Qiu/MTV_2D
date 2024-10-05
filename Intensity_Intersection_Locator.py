import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks

image_location = "Synthetic_Data/SNR_2/Set_0/Rotational_Flow_Image_Set_0.png"
# image_location = "Template_Matching_Test/Source/frame_2.png"
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(image, (5, 5), 0).astype('uint8')

# Apply adaptive thresholding and Otsu's thresholding
adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
ret, ot = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to Eliminate Noises
kernel = np.ones((3, 3), np.uint8)
adaptive_thresh = cv2.erode(adaptive_thresh, kernel=kernel)
adaptive_thresh = cv2.dilate(adaptive_thresh, kernel=kernel)

# Create a gradient mask to isolate specific regions
grad_x = cv2.Sobel(ot, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(ot, cv2.CV_64F, 0, 1, ksize=5)

grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
gradient_mask = (grad_magnitude == 0)
dark_mask = (ot == 0)
mask = gradient_mask & dark_mask

img_copy = image.copy()
img_copy[mask] = 0

adaptive_thresh[mask] = 0

# Apply skeletonization
skeleton = skeletonize(adaptive_thresh).astype(np.uint8)

# Calculate Sobel gradients
grad_x_skel = cv2.Sobel(skeleton, cv2.CV_64F, 1, 0, ksize=5)
grad_y_skel = cv2.Sobel(skeleton, cv2.CV_64F, 0, 1, ksize=5)
grad_xy_skel = cv2.Sobel(grad_x_skel, cv2.CV_64F, 0, 1, ksize=5)

"""
Gradient across sobel_xy must be large for the existence of an
intersection. Implemented algorithm should take into cosideration
the possibility of multiple high gradient --> Use a window proportional
to the half width of the cross section. --> Could co-implement with line
matching to reduce outliers by bounding location of intersection. Or 
utlize a gradient threshold to be consider an intersection point. 
"""

plt.figure(figsize=(8, 6))
plt.title("Sobel x")
plt.imshow(grad_x_skel, cmap=cm.gray)
plt.colorbar()

plt.figure(figsize=(8, 6))
plt.title("Sobel y")
plt.imshow(grad_y_skel, cmap=cm.gray)
plt.colorbar()

plt.figure(figsize=(8, 6))
plt.title("Sobel xy")
plt.imshow(grad_xy_skel, cmap=cm.gray)
plt.colorbar()
plt.show()

# Get the absolute values of the gradients
abs_grad_x = np.abs(grad_x_skel)
abs_grad_y = np.abs(grad_y_skel)


# Function to get non-overlapping maximum gradients in a 5x5 window
def get_max_gradients(abs_grad, window_size=5, max_count=121):
    max_locations = []
    max_values = []

    rows, cols = abs_grad.shape

    # Slide the window over the image
    for i in range(0, rows - window_size + 1, window_size):
        for j in range(0, cols - window_size + 1, window_size):
            window = abs_grad[i:i + window_size, j:j + window_size]
            max_value = np.max(window)
            max_index = np.unravel_index(np.argmax(window), window.shape)
            max_locations.append((i + max_index[0], j + max_index[1]))  # Store the location of max value
            max_values.append(max_value)

    # Sort by max values and get the top max_count
    sorted_indices = np.argsort(max_values)[-max_count:]
    return np.array(max_locations)[sorted_indices]


# Get the locations of the largest gradients
max_grad_x_locations = get_max_gradients(abs_grad_x)
max_grad_y_locations = get_max_gradients(abs_grad_y)

# Create a copy of the skeleton for drawing circles
skeleton_with_circles = cv2.cvtColor(skeleton * 255, cv2.COLOR_GRAY2BGR)

# Draw circles on the locations of largest gradients
for loc in max_grad_x_locations:
    cv2.circle(skeleton_with_circles, (loc[1], loc[0]), 5, (255, 0, 0), 2)  # Red for x gradient

for loc in max_grad_y_locations:
    cv2.circle(skeleton_with_circles, (loc[1], loc[0]), 5, (0, 255, 0), 2)  # Green for y gradient

# Plot the results
plt.figure(figsize=(10, 10))
plt.imshow(skeleton_with_circles)
plt.title('Skeleton with Non-overlapping Largest Sobel Gradients')
plt.axis('off')
plt.show()

# Show the images
fig, axes = plt.subplots(2, 3, figsize=(20, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(adaptive_thresh, cmap=cm.gray)
ax[1].set_title('Binary')
ax[1].set_axis_off()

ax[2].imshow(skeleton, cmap=cm.gray)
ax[2].set_title('Skeleton Image')
ax[2].set_axis_off()

# Detected lines plot
ax[3].imshow(image, cmap=cm.gray)
ax[3].set_ylim((image.shape[0], 0))
ax[3].set_axis_off()
ax[3].set_title('Detected lines')

ax[4].imshow(image, cmap=cm.gray)
ax[4].set_ylim((image.shape[0], 0))
ax[4].set_axis_off()
ax[4].set_title('Detected Intersections')

ax[5].imshow(image, cmap=cm.gray)
ax[5].set_ylim((image.shape[0], 0))
ax[5].set_axis_off()
ax[5].set_title('Sub Pixel Intersections')

# Hough transform parameters
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
h, theta, d = hough_line(skeleton, theta=tested_angles)
height, width = image.shape[:2]

# Store line parameters (angle and distance)
lines = []

# Extract lines using hough_line_peaks
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope = np.tan(angle + np.pi / 2)
    lines.append((angle, dist))
    borders = [(0, 0), (width, 0), (0, height), (width, height)]
    x_left, x_right = 0, width
    y_left = slope * (x_left - x0) + y0
    y_right = slope * (x_right - x0) + y0
    y_top, y_bottom = 0, height
    x_top = (y_top - y0) / slope + x0 if slope != 0 else np.inf
    x_bottom = (y_bottom - y0) / slope + x0 if slope != 0 else np.inf

    points = [(x_left, y_left), (x_right, y_right), (x_top, y_top), (x_bottom, y_bottom)]
    points_in_image = [(x, y) for x, y in points if 0 <= x <= width and 0 <= y <= height]

    if len(points_in_image) >= 2:
        points_in_image.sort()
        ax[3].plot([points_in_image[0][0], points_in_image[1][0]],
                   [points_in_image[0][1], points_in_image[1][1]], color='red')

# Calculate line intersections
intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
        # Line 1: a1*x + b1*y = c1
        # Line 2: a2*x + b2*y = c2
        angle1, dist1 = lines[i]
        angle2, dist2 = lines[j]

        # Calculate coefficients for the two lines
        a1, b1 = np.cos(angle1), np.sin(angle1)
        a2, b2 = np.cos(angle2), np.sin(angle2)
        c1, c2 = dist1, dist2

        # Calculate determinant
        determinant = a1 * b2 - a2 * b1

        if np.abs(determinant) < 1e-10:
            # Lines are parallel or too close to being parallel
            continue

        # Calculate intersection using Cramer's rule
        x = (c1 * b2 - c2 * b1) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        # Check if intersection is within image bounds
        if 0 <= x <= width and 0 <= y <= height:
            intersections.append((x, y))

# Plot the intersections as green dots
intersections = np.array(intersections, dtype=np.int32).astype(np.float32)
ax[4].scatter(intersections[:, 0], intersections[:, 1], color='green', s=5, label='Intersections')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

if intersections.size > 0:
    # Reshape to (N, 1, 2) format required by cornerSubPix
    intersections.reshape(-1, 1, 2)  # Use -1 to handle any number of points dynamically
    sub_pixel_corners = cv2.cornerSubPix(image, intersections, winSize=(5, 5),
                                         zeroZone=(-1, -1), criteria=criteria)

    ax[5].scatter(np.intp(sub_pixel_corners[:, 0]), np.intp(sub_pixel_corners[:, 1]),
                  color='green', s=5, label='Intersections')

# Show the plot
plt.tight_layout()
plt.show()
