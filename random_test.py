import cv2
import os
import numpy as np
import glob

# --------- USER SETTINGS ----------
video_path = 'DJI_0025_Trim_face2.mp4'  # <-- Place your drone video file here
output_frames_dir = 'drone_frames'
frame_interval = 60   # Extract every Nth frame, adjust depending on drone speed
min_required_matches = 8  # Require more matches for higher stitching reliability
resize_output = (1280, 720)  # Final panorama display size
# -----------------------------------

os.makedirs(output_frames_dir, exist_ok=True)

# 1. Extract Frames from Drone Footage
cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total frames in video: {total_frames}")
print(f"Extracting every {frame_interval}th frame...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_path = os.path.join(output_frames_dir, f"frame_{saved_count:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames to {output_frames_dir}")

# 2. Load Frames
image_paths = sorted(glob.glob(f"{output_frames_dir}/*.jpg"))
if len(image_paths) < 2:
    print("Not enough frames extracted. Try lowering frame_interval.")
    exit()

def create_image_list(image_paths):
    return [cv2.imread(image) for image in image_paths]

images = create_image_list(image_paths)

# 3. ASIFT Functions
def affine_warp(img, tilt, rotation):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M_tilt = np.array([[1, 0, 0], [0, 1 / tilt, 0]], dtype=np.float32)
    img_tilt = cv2.warpAffine(img, M_tilt, (w, h))
    M_rotate = cv2.getRotationMatrix2D(center, rotation, 1.0)
    img_affine = cv2.warpAffine(img_tilt, M_rotate, (w, h))
    return img_affine

def detect_asift_features(img, tilt_list=[1, 1.5, 2.0], rot_list=[0, 45, 90, 135, 180]):
    sift = cv2.SIFT_create()
    keypoints, descriptors = [], []
    for tilt in tilt_list:
        for rot in rot_list:
            warped = affine_warp(img, tilt, rot)
            kp, des = sift.detectAndCompute(warped, None)
            if kp and des is not None:
                for k in kp:
                    k.pt = (k.pt[0], k.pt[1] * tilt)
                keypoints.extend(kp)
                descriptors.append(des)
    return keypoints, np.vstack(descriptors) if descriptors else None

def trim(frame):
    mask = np.any(frame > 0, axis=2) if len(frame.shape) == 3 else frame > 0
    coords = np.argwhere(mask)
    if coords.size == 0: return frame
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return frame[y0:y1, x0:x1]

def stitch_asift(img1, img2):
    kp1, des1 = detect_asift_features(img1)
    kp2, des2 = detect_asift_features(img2)
    if des1 is None or des2 is None:
        print("No descriptors found in one of the images. Skipping.")
        return img1

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > min_required_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        corners_img2 = np.float32([[0,0], [0,h2], [w2,h2], [w2,0]]).reshape(-1,1,2)
        transformed_corners = cv2.perspectiveTransform(corners_img2, M)
        corners_img1 = np.float32([[0,0], [0,h1], [w1,h1], [w1,0]]).reshape(-1,1,2)
        all_corners = np.concatenate((transformed_corners, corners_img1), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        translation_dist = [-xmin, -ymin]
        H_translation = np.array([[1, 0, translation_dist[0]],
                                  [0, 1, translation_dist[1]],
                                  [0, 0, 1]])
        result = cv2.warpPerspective(img2, H_translation @ M, (xmax - xmin, ymax - ymin))
        result[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = img1
        return trim(result)
    else:
        print(f"Not enough good matches: {len(good)}. Skipping.")
        return img1

# 4. Build Panorama
print(f"Stitching {len(images)} frames into panorama...")
panorama = images[0]

for i in range(1, len(images)):
    next_img = images[i]
    stitched = stitch_asift(panorama, next_img)
    if stitched.shape != panorama.shape:
        panorama = stitched
    else:
        print(f"Skipping frame {i}, stitching unchanged.")

# 5. Final Output
panorama_resized = cv2.resize(panorama, resize_output)
cv2.imshow('Drone Panorama', panorama_resized)
cv2.imwrite('drone_panorama.jpg', panorama_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Panorama saved as 'drone_panorama.jpg'. Done!")
