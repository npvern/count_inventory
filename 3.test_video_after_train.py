import cv2
from ultralytics import YOLO
from datetime import datetime
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

now = datetime.now()
model = YOLO("runs/obb/train14/weights/best.pt")
cap = cv2.VideoCapture(r"DJI_0025_Trim_face2.mp4")
# timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# original_fps = cap.get(cv2.CAP_PROP_FPS)
# video_out = cv2.VideoWriter('video_out/{}.avi'.format(timestamp),fourcc, 25.0, (1280,720))
result_set = set()
number_of_bags = 0
fps_start_time = 0
fps = 0

def display_fps(frame, fps_start_time):
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff if time_diff > 0 else 0  # ป้องกันการหารด้วยศูนย์
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    return fps_end_time  # ส่งค่าเวลาปัจจุบันกลับเพื่อให้นำไปใช้ในการคำนวณ FPS ในเฟรมถัดไป

# tracker = DeepSort(max_iou_distance=0.7, n_init=3, max_age=5, n_features=10, embedder="mobilenet")

while cap.isOpened():
    # Read a frame from the video
    ret, img1 = cap.read()
    img = cv2.resize(img1, (1280, 720))
    if ret is False:
        break
        # detections = your_object_detector.detect(frame)
    

    results = model(source=img, conf=0.5, visualize=False, imgsz=1280)
    result = results[0]
    if result.obb is not None:
        result_set.add(result[0])
        number_of_bags = len(result.obb)

    annotated_frame = results[0].plot(conf=True, line_width=None, font_size=None, font='Arial.ttf', 
                                    pil=False, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True,
                                    boxes=True, masks=True, probs=True, show=False, save=False, filename=None) #ซ้อน labels และ กรอบ
    fps_start_time = display_fps(annotated_frame, fps_start_time)

    frame_height, frame_width = annotated_frame.shape[:2]

    cv2.putText(annotated_frame, f'Number of Bags: {number_of_bags}', (900, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0),2 )
    cv2.imshow('Bigbag', annotated_frame)
    # video_out.write(annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()               
# video_out.release()    
cv2.destroyAllWindows()
