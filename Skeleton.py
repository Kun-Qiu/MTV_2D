import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize, remove_small_objects

# image_location = "Deep Learning/TIFF/bp0_m0p2_2microsec.tif_76.png"
image_location = "Template_Matching_Test/Source/frame_2.png"
# image_location = "Template_Matching_Test/Target/frame_2_2us.png"
# image_location = "Synthetic_Data/SNR_1/Set_0/Rotational_Flow_Image_Set_0.png"
# image_location = "Template_Matching_Test/Source/source_avg.png"
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
skeleton_before = skeletonize(adaptive_thresh // 255).astype(np.uint8)

# Label the skeleton (regions must be labeled for `remove_small_objects`)
skeleton_labeled = label(skeleton_before)
skeleton_small_removed = remove_small_objects(skeleton_labeled, min_size=10)

# Convert labeled array back to binary (Boolean) format
skeleton = (skeleton_small_removed > 0).astype(np.uint8)

# ------------------------------
# Step 6: Intersection Detection
# ------------------------------

# Initialize a list to hold intersection points
intersections = []

# Get the dimensions of the skeleton image
rows, cols = skeleton.shape

# Define the relative positions of the 8 neighbors (clockwise order)
# Window size of 3x3
neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                    (1, 1), (1, 0), (1, -1), (0, -1)]

# Iterate over each pixel in the skeleton (excluding the border pixels)
for y in range(1, rows - 1):
    for x in range(1, cols - 1):
        if skeleton[y, x]:
            # Extract the 8-connected neighbors
            neighbors = [skeleton[y + dy, x + dx] for dy, dx in neighbor_offsets]

            # Count the number of neighbors with value 1
            num_neighbors = sum(neighbors)

            # Count the number of rises (0 to 1 transitions) in the neighborhood
            rises = 0
            for i in range(len(neighbor_offsets)):
                if neighbors[i] == 0 and neighbors[(i + 1) % len(neighbor_offsets)] == 1:
                    rises += 1

            # Apply the intersection criteria
            if (rises >= 3 and num_neighbors >= 3) or (rises > 1 and num_neighbors > 4):
                intersections.append((x, y))
                # Optionally, mark the intersection on the skeleton
                skeleton[y, x] = 2  # Using value 2 to differentiate

# ------------------------------
# Step 7: Visualization
# ------------------------------

# Convert the skeleton to a color image for visualization
skeleton_color = cv2.cvtColor((skeleton * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Draw red circles on the intersection points
for (x, y) in intersections:
    cv2.circle(skeleton_color, (x, y), 1, (0, 0, 255), -1)  # Red color

# Display the skeleton and intersections using matplotlib
plt.figure(figsize=(12, 6))

# Display Skeleton
plt.subplot(1, 3, 1)
plt.title('Skeleton of the Grid')
plt.imshow(skeleton_before, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Cleaned Skeleton')
plt.imshow(skeleton)
plt.axis('off')

# Display Skeleton with Intersections
plt.subplot(1, 3, 3)
plt.title('Intersections Detected')
plt.imshow(skeleton_color)
plt.axis('off')

plt.tight_layout()
plt.show()

# Print the total number of intersections found
print(f"Total intersections found: {len(intersections)}")
