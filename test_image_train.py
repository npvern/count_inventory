import cv2
from ultralytics import YOLO
from datetime import datetime
now = datetime.now()

model = YOLO("runs/obb/train14/weights/best.pt")
image = cv2.imread("combined_warped.jpg")
results = model(source=image, conf=0.6, visualize=False, imgsz=1280)

result = results[0]

if result.obb is not None:
    number_of_bags = len(result.obb)
else:
    number_of_bags = 0

for result in results:
    annotated_image = result.plot(conf=True, line_width=None, font_size=None, font='Arial.ttf', 
                                    pil=False, img=None, im_gpu=None, kpt_radius=5, kpt_line=True, labels=True,
                                    boxes=True, masks=True, probs=True, show=False, save=False, filename=None)
   
frame_height, frame_width = annotated_image.shape[:2]

cv2.putText(annotated_image, f'Number of Bags: {number_of_bags}', (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2 )

cv2.imwrite('counted_bags.jpg', annotated_image) 
cv2.imshow('BigBag', annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows




