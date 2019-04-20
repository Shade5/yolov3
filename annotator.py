import cv2
import numpy as np
from glob import glob


def onMouse(event, x, y, flags, param):
	global count
	if event == cv2.EVENT_LBUTTONDOWN:
		print('x = %d, y = %d'%(x, y))
		clicks.append((x, y))
		count += 1


cv2.namedWindow('a')
cv2.setMouseCallback('a', onMouse)

data = {}

for im_path in glob("/home/a/Desktop/bar/*"):
	count = 0
	clicks = []
	im = cv2.resize(cv2.imread(im_path), (416, 416))
	cv2.imshow('a', im)
	while count < 2:
		cv2.waitKey(10)

	data[im_path] = ['bar', 0, [[clicks[0][1], clicks[0][0], clicks[1][1] - clicks[0][1], clicks[1][0] - clicks[0][0]]]]

	(noun, noun_class, bbox) = data[im_path]
	for y, x, he, we in bbox:
		im = cv2.rectangle(im, (x, y), (x+we, y+he), (0, 0, 255), 2)
	print(data)
	cv2.imshow('a', im)
	cv2.waitKey(0)
