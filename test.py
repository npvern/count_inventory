# import cv2
# import numpy as np
# import glob

# image_paths = sorted(glob.glob('video_to_stitch2/*.jpg'))
# images = []

# for image in image_paths:
#     number_frames = len(image_paths)
#     img = cv2.imread(image)
#     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#     images.append(img)

# imageStitcher = cv2.Stitcher_create()

# warped_img = []
# for warp in images:
# # Define 4 points in the source image
#     pts_src = np.float32([[100, 100],
#                         [400, 100],
#                         [400, 400],
#                         [100, 400]])

#     # Define where you want those points to go
#     pts_dst = np.float32([[0, 0],
#                         [300, 0],
#                         [300, 300],
#                         [0, 300]])

#     # Compute homography matrix
#     H, status = cv2.findHomography(pts_src, pts_dst)

#     # Warp the perspective using the homography
#     warped = cv2.warpPerspective(warp, H, (720, 1280))
#     warped_img.append(warped)

# error, stitched_img = imageStitcher.stitch(warped_img)

# # Show output
# if not error:
#     resized_stitched_img = cv2.resize(stitched_img, (1280, 720))
#     cv2.imwrite('stitchedOutput.jpg', resized_stitched_img)
#     cv2.imshow('stitched Image', resized_stitched_img)
#     cv2.waitKey(0)
# else:
#     print("Stitching failed. Error code:", error)


import cv2
import numpy as np
import glob

# Get and sort image paths
image_paths = sorted(glob.glob('video_to_stitch2/*.jpg'))
images = []

# Read and rotate images
for path in image_paths:
    img = cv2.imread(path)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    images.append(img)

# Initialize ORB detector
orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Use the first image as the base
stitched_img = images[0]

for i in range(1, len(images)):
    img_next = images[i]

    # Detect features and descriptors
    kp1, des1 = orb.detectAndCompute(stitched_img, None)
    kp2, des2 = orb.detectAndCompute(img_next, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print(f"Skipping frame {i} due to insufficient keypoints.")
        continue

    # Match descriptors
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print(f"Skipping frame {i} due to insufficient matches.")
        continue

    # Get matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute Homography
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        print(f"Skipping frame {i}, Homography failed.")
        continue

    # Warp next image to align with stitched_img
    h1, w1 = stitched_img.shape[:2]
    h2, w2 = img_next.shape[:2]

    # Calculate size of final stitched image
    corners_img_next = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img_next, H)

    all_corners = np.concatenate((warped_corners, np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation[0]],
                              [0, 1, translation[1]],
                              [0, 0, 1]])

    result_img = cv2.warpPerspective(img_next, H_translation @ H, (xmax - xmin, ymax - ymin))
    result_img[translation[1]:translation[1]+h1, translation[0]:translation[0]+w1] = stitched_img

    stitched_img = result_img

# Show and save final stitched image
cv2.imwrite('stitchedOutput1.jpg', stitched_img)
cv2.imshow('Stitched Image', stitched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
