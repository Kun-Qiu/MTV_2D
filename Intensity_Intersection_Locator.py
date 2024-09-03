import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

# Load the image in grayscale
image_location = "Template_Matching_Test/Target/frame_2_2us.png"
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

# Step 1: Adaptive Thresholding
blur = cv2.GaussianBlur(image, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply skeletonization
skeleton = skeletonize(th3 // 255).astype(np.uint8) * 255

# Apply Probabilistic Hough Line Transform
minLineLength = 50
maxLineGap = 15
threshold = 30
lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=threshold,
                        minLineLength=minLineLength, maxLineGap=maxLineGap)

og_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(og_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

"""
Calculate Slope Intercept of Line Segments
"""
threshold = 0.1

line_np = lines[:, 0]
slope_line = (line_np[:, 3] - line_np[:, 1]) / (line_np[:, 2] - line_np[:, 0])
intercept = line_np[:, 1] - slope_line * line_np[:, 0]
index_positive = slope_line > 0
index_negative = slope_line < 0

features = np.hstack([slope_line[:, np.newaxis], intercept[:, np.newaxis]])

# Apply DBSCAN clustering to group lines with similar slopes
clustering = DBSCAN(eps=threshold, min_samples=1).fit(features)
labels = clustering.labels_

# Compute the average slope and intercept for each cluster
unique_labels = np.unique(labels)
average_lines = []

for label in unique_labels:
    indices = np.where(labels == label)[0]
    avg_slope = np.mean(slope_line[indices])
    avg_intercept = np.mean(intercept[indices])
    average_lines.append([avg_slope, avg_intercept])

# Convert the result to a numpy array for further processing
average_lines = np.array(average_lines)

# Create a color image to draw lines on
color_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)

# Draw the merged lines on the image
for line in merged_lines:
    x1, y1, x2, y2 = line
    cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Skeleton Image')
plt.imshow(skeleton, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Original')
plt.imshow(cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Combined')
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

#
# # Step 2: Harris Corner Detection
# skeleton_float = np.float32(skeleton)
# dst = cv2.cornerHarris(skeleton_float, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
#
# # Apply threshold to get the corners
# ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
# dst = np.uint8(dst)
#
# # Find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#
# # Define criteria to refine corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# corners = cv2.cornerSubPix(np.float32(image), np.float32(centroids), (5, 5), (-1, -1), criteria)
#
# # Convert results to correct format
# corners = np.array(corners, dtype=np.float32)
# corners = np.squeeze(corners)
#
# # Create color images for visualization
# skeleton_color = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
# image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
# # Mark the refined corners on the images
# for corner in corners:
#     x, y = np.int0(corner)
#     cv2.circle(skeleton_color, (x, y), 2, (0, 0, 255), -1)  # Red circle for corners
#     cv2.circle(image_color, (x, y), 2, (0, 0, 255), -1)  # Red circle for corners
#
# # Convert BGR to RGB for Matplotlib
# skeleton_rgb = cv2.cvtColor(skeleton_color, cv2.COLOR_BGR2RGB)
# image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
#
# # Display the images with detected corners using Matplotlib
# plt.figure(figsize=(12, 6))
#
# # Plot original image with corners
# plt.subplot(1, 2, 1)
# plt.imshow(image_rgb)
# plt.title('Detected Corners in Original Image')
# plt.axis('off')
#
# # Plot skeleton image with corners
# plt.subplot(1, 2, 2)
# plt.imshow(skeleton_rgb)
# plt.title('Detected Corners in Skeleton Image')
# plt.axis('off')
#
# plt.show()
