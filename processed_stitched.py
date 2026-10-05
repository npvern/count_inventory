import cv2
import numpy as np
import imutils

stitched = cv2.imread('stitchedOutput0.jpg')
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

cv2.imshow("Threshold Image", thresh_img)
cv2.waitKey(0)

contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)

peri = cv2.arcLength(areaOI, True)  # contour perimeter
approx = cv2.approxPolyDP(areaOI, 0.02 * peri, True)

cv2.drawContours(thresh_img, [approx], -1, (0, 255, 0), 2)
cv2.imshow("Threshold Image", thresh_img)
cv2.waitKey(0)

src_pts = approx.reshape(-1, 2).astype(np.float32)

height, width = 1280, 720
dst_pts = np.float32([
    [0, 0],             # top-left
    [width, 0],     # top-right
    [width, height],  # bottom-right
    [0, height]     # bottom-left
])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)

warped = cv2.warpPerspective(stitched, M, (width, height))
warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

img_height, img_width, _ = warped.shape
 
warped = warped[100:img_height-100, 0:img_width]

cv2.imshow('warped stitched', warped)
cv2.imwrite('warped_stitched.jpg', warped)
cv2.waitKey(0)