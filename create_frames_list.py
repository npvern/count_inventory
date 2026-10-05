# import cv2
# import os

# video_path = 'DJI_0025_Trim_face2.mp4'
# output_folder = 'video_to_stitch'
# frame_count = 0
# saved_count = 0
# capture_interval_ms = 1000  # Capture every 500 milliseconds

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# video = cv2.VideoCapture(video_path)
# if not video.isOpened():
#     print("Error opening video file")
#     exit()

# fps = video.get(cv2.CAP_PROP_FPS)
# frame_interval = int((capture_interval_ms / 1000) * fps)

# print(f"Video FPS: {fps}, capturing every {frame_interval} frames")

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     if frame_count % frame_interval == 0:
#         frame_name = f"frame_{saved_count:04d}.jpg"
#         frame_path = os.path.join(output_folder, frame_name)
#         cv2.imwrite(frame_path, frame)
#         saved_count += 1

#     frame_count += 1

# video.release()
# print(f"Successfully saved {saved_count} frames to '{output_folder}'")

import cv2
import os

video_path = 'DJI_0025_Trim_face2.mp4'
output_folder = 'video_to_stitch2'
frame_count = 0
saved_count = 0
capture_interval_ms = 1000  # Capture every 1000 milliseconds (1 second)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error opening video file")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
halfway_point = total_frames // 2
frame_interval = int((capture_interval_ms / 1000) * fps)

print(f"Video FPS: {fps}, capturing every {frame_interval} frames")
print(f"Capturing up to frame {halfway_point} (halfway point)")

while frame_count < halfway_point:
    ret, frame = video.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_name = f"frame_{saved_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    frame_count += 1

video.release()
print(f"Successfully saved {saved_count} frames to '{output_folder}'")
