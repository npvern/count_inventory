import cv2
import numpy as np
import glob
import imutils
import torch

image_paths = glob.glob('video_to_stitch2/*.jpg')
images = []

for image in image_paths:
    number_frames = len(image_paths)
    img = cv2.imread(image)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    images.append(img)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)

imageStitcher = cv2.Stitcher_create()

error, stitched_img = imageStitcher.stitch(images)

if not error:
    resized_stitched_img = cv2.resize(stitched_img, (1280, 720))
    cv2.imwrite('stitchedOutput0.jpg', resized_stitched_img)
    cv2.imshow('stitched Image', resized_stitched_img)
    cv2.waitKey(0)
else:
    print("Stitching failed. Error code:", error)

