# import cv2
# import numpy as np
# import imutils
# import matplotlib.pyplot as plt

# stitched = cv2.imread('stitchedOutput0.jpg')
# stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
# gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
# thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

# cv2.imshow("Threshold Image", thresh_img)
# cv2.waitKey(0)

# contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contours = imutils.grab_contours(contours)
# areaOI = max(contours, key=cv2.contourArea)

# peri = cv2.arcLength(areaOI, True)  # contour perimeter
# approx = cv2.approxPolyDP(areaOI, 0.02 * peri, True)

# cv2.drawContours(thresh_img, [approx], -1, (0, 255, 0), 2)
# cv2.imshow("Threshold Image", thresh_img)
# cv2.waitKey(0)

# src_pts = approx.reshape(-1, 2).astype(np.float32)

# height, width = 1280, 720
# dst_pts = np.float32([
#     [0, 0],             # top-left
#     [width, 0],     # top-right
#     [width, height],  # bottom-right
#     [0, height]     # bottom-left
# ])

# M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# warped = cv2.warpPerspective(stitched, M, (width, height))
# warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

# cv2.imshow('warped stitched', warped)
# cv2.imwrite('warped_stitched.jpg', warped)
# cv2.waitKey(0)

# points = areaOI.reshape(-1, 2)

# leftmost   = tuple(points[points[:, 0].argmin()])
# rightmost  = tuple(points[points[:, 0].argmax()])
# topmost    = tuple(points[points[:, 1].argmin()])
# bottommost = tuple(points[points[:, 1].argmax()])

# thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)

# for pt in points:
#     cv2.circle(thresh_img, tuple(pt), 3, (0, 255, 0), -1)

# cv2.imshow('thresh', thresh_img)
# cv2.waitKey(0)

# gray = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)

# # Threshold and edge detection
# _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# edges = cv2.Canny(thresh, 50, 150)

# # Hough Line Transform
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# # Filter horizontal lines
# horizontal_lines = []
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     if abs(y2 - y1) < 10:  # nearly horizontal
#         horizontal_lines.append((x1, y1, x2, y2))

# # Sort horizontal lines by y-coordinate
# horizontal_lines = sorted(horizontal_lines, key=lambda l: l[1])

# # Get top edge line
# top_line = horizontal_lines[0]
# x1, y1, x2, y2 = top_line

# # Analyze slope changes along the top line
# start_x = int(min(x1, x2) + 0.15 * abs(x2 - x1))
# end_x = max(x1, x2)

# segment_len = 10
# slope_changes = []
# for i in range(start_x, end_x - segment_len, segment_len):
#     x_start = i
#     x_end = i + segment_len
#     y_start = int(np.interp(x_start, [x1, x2], [y1, y2]))
#     y_end = int(np.interp(x_end, [x1, x2], [y1, y2]))
#     slope = (y_end - y_start) / (x_end - x_start + 1e-5)
#     slope_changes.append((x_start, slope))

# # Detect sharpest slope change
# slope_deltas = [abs(s2 - s1) for (_, s1), (_, s2) in zip(slope_changes[:-1], slope_changes[1:])]
# split_index = np.argmax(slope_deltas)
# split_x = slope_changes[split_index][0]

# # Draw top horizontal line and split line on the original image before cropping
# visual = thresh_img.copy()
# cv2.line(visual, (x1, y1), (x2, y2), (255, 0, 0), 2)  # top horizontal line (blue)
# cv2.line(visual, (split_x, 0), (split_x, visual.shape[0]), (0, 0, 255), 2)  # split line (red)

# # Show image with the lines
# visual_rgb = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(12, 6))
# plt.imshow(visual_rgb)
# plt.title("Detected Top Line (Blue) and Split Position (Red)")
# plt.axis("off")
# plt.show()

# # Now crop
# left_crop = thresh_img[:, :split_x]
# right_crop = thresh_img[:, split_x:]

# # Show both sections side-by-side
# left_rgb = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)
# right_rgb = cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# axs[0].imshow(left_rgb)
# axs[0].set_title("Left: Before Shrinkage")
# axs[0].axis("off")

# axs[1].imshow(right_rgb)
# axs[1].set_title("Right: After Shrinkage")
# axs[1].axis("off")

# plt.tight_layout()
# plt.show()

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

# Load the image
stitched = cv2.imread('stitchedOutput0.jpg')
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
_, thresh_img = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours and the largest one
contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)
points = areaOI.reshape(-1, 2)

# Image dimensions
image_height, image_width = stitched.shape[:2]

# Keep only points from top 60% of the image and exclude leftmost 15%
top_points = points[(points[:, 1] < image_height * 0.6) & (points[:, 0] > image_width * 0.15)]

# Get leftmost and rightmost x among filtered points
leftmost = tuple(top_points[top_points[:, 0].argmin()])
rightmost = tuple(top_points[top_points[:, 0].argmax()])

# Analyze slope along x between leftmost and rightmost
segment_len = 10
slope_changes = []
x_vals = np.linspace(leftmost[0], rightmost[0], num=(rightmost[0] - leftmost[0]) // segment_len, dtype=int)

for i in range(len(x_vals) - 1):
    x_start = x_vals[i]
    x_end = x_vals[i + 1]

    y_start = np.median(top_points[(top_points[:, 0] >= x_start) & (top_points[:, 0] < x_start + segment_len), 1])
    y_end = np.median(top_points[(top_points[:, 0] >= x_end) & (top_points[:, 0] < x_end + segment_len), 1])

    if not np.isnan(y_start) and not np.isnan(y_end):
        slope = (y_end - y_start) / (x_end - x_start + 1e-5)
        slope_changes.append((x_start, slope))

# Find sharpest slope change
slope_deltas = [abs(s2 - s1) for (_, s1), (_, s2) in zip(slope_changes[:-1], slope_changes[1:])]
split_index = np.argmax(slope_deltas)
split_x = int(slope_changes[split_index][0])

# Draw points and split line
visual = stitched.copy()
for pt in top_points:
    cv2.circle(visual, tuple(pt), 2, (0, 255, 0), -1)
cv2.imshow("Top Points Overlay", visual)
cv2.waitKey(0)

cv2.line(visual, (split_x, 0), (split_x, image_height), (0, 0, 255), 2)

# Crop the image into left and right sections
left_crop = stitched[:, :split_x]
right_crop = stitched[:, split_x:]

# Convert to RGB for matplotlib
visual_rgb = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)
left_rgb = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)

# Display the result
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(visual_rgb)
axs[0].set_title("Top Points & Split Line")
axs[0].axis("off")

axs[1].imshow(left_rgb)
axs[1].set_title("Left: Before Shrinkage")
axs[1].axis("off")

axs[2].imshow(right_rgb)
axs[2].set_title("Right: After Shrinkage")
axs[2].axis("off")

plt.tight_layout()
plt.show()

h_left, w_left = left_crop.shape[:2]
h_right, w_right = right_crop.shape[:2]

left_crop_gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
right_crop_gray = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)

cv2.imshow('left', left_crop_gray)
cv2.waitKey(0)
cv2.imshow('right', right_crop_gray)
cv2.waitKey(0)

thresh_img_left = cv2.threshold(left_crop_gray, 0, 255 , cv2.THRESH_BINARY)[1]
thresh_img_right = cv2.threshold(right_crop_gray, 0, 255 , cv2.THRESH_BINARY)[1]

contours_left = cv2.findContours(thresh_img_left.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_left = imutils.grab_contours(contours_left)
areaOI_left = max(contours_left, key=cv2.contourArea)

cv2.imshow("Threshold Image", thresh_img_left)
cv2.waitKey(0)
cv2.imshow("Threshold Image", thresh_img_right)
cv2.waitKey(0)

peri_left = cv2.arcLength(areaOI_left, True)  # contour perimeter
approx_left = cv2.approxPolyDP(areaOI_left, 0.02 * peri_left, True)

contours_right = cv2.findContours(thresh_img_right.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours_right = imutils.grab_contours(contours_right)
areaOI_right = max(contours_right, key=cv2.contourArea)

peri_right = cv2.arcLength(areaOI_right, True)  # contour perimeter
approx_right = cv2.approxPolyDP(areaOI_right, 0.02 * peri_right, True)

cv2.drawContours(thresh_img_left, [approx_left], -1, (0, 255, 0), 2)
cv2.drawContours(thresh_img_right, [approx_right], -1, (0, 255, 0), 2)

cv2.imshow("Threshold Image", thresh_img_left)
cv2.waitKey(0)
cv2.imshow("Threshold Image", thresh_img_right)
cv2.waitKey(0)

left_pts = approx_left.reshape(-1, 2).astype(np.float32)
right_pts = approx_right.reshape(-1, 2).astype(np.float32)

height, width = 1280, 720
dst_pts = np.float32([
    [0, 0],             # top-left
    [width, 0],     # top-right
    [width, height],  # bottom-right
    [0, height]     # bottom-left
])

M_left = cv2.getPerspectiveTransform(left_pts, dst_pts)
warped_left = cv2.warpPerspective(left_crop, M_left, (720, 1280))
warped_left = cv2.rotate(warped_left, cv2.ROTATE_90_COUNTERCLOCKWISE)

M_right = cv2.getPerspectiveTransform(right_pts, dst_pts)
warped_right = cv2.warpPerspective(right_crop, M_right, (720, 1280))
warped_right = cv2.rotate(warped_right, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Show both warped halves
cv2.imshow("Warped Left (Before Shrinkage)", warped_left)
cv2.imshow("Warped Right (After Shrinkage)", warped_right)

cv2.imwrite("warped_left.jpg", warped_left)
cv2.imwrite("warped_right.jpg", warped_right)

# Ensure both warped sides have the same height
target_height = 720
target_width = 1280

warped_left_resized = cv2.resize(warped_left, (target_width, target_height))
warped_right_resized = cv2.resize(warped_right, (target_width, target_height))

# Combine them horizontally
combined = np.hstack([warped_left_resized, warped_right_resized])
combined = cv2.resize(combined, (target_width, target_height))
# Alternatively: combined = cv2.hconcat([warped_left_resized, warped_right_resized])

# Show and save result
img_height, img_width, _ = combined.shape

# combined = combined[100:img_height-100, 0:img_width]

cv2.imshow("Combined Warped Image", combined)
cv2.imwrite("combined_warped.jpg", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

