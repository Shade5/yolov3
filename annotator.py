import cv2
import numpy as np
from glob import glob
import os
import pickle
from tqdm import tqdm


def onMouse(event, x, y, flags, param):
	global count
	if event == cv2.EVENT_LBUTTONDOWN:
		# print('x = %d, y = %d'%(x, y))
		clicks.append((x, y))
		count += 1


cv2.namedWindow('a')
cv2.setMouseCallback('a', onMouse)


with open('boxes_fixed.pickle', 'rb') as handle:
	data = pickle.load(handle)

for i, im_path in enumerate(glob("/home/a/Desktop/bar/*")):
	count = 0
	clicks = []
	im = cv2.resize(cv2.imread(im_path), (416, 416))
	cv2.imshow('a', im)

	entries = data[im_path]
	for (noun, noun_class, bbox) in entries:
		for y, x, he, we in bbox:
			im = cv2.rectangle(im, (x, y), (x+we, y+he), (0, 0, 255), 2)
	print(data)
	cv2.imshow('a', im)
	cv2.waitKey(0)

########################################################################

# data = {}
#
# for i, im_path in enumerate(tqdm(glob("/home/a/Desktop/bar/*"))):
# 	count = 0
# 	clicks = []
# 	im = cv2.resize(cv2.imread(im_path), (416, 416))
# 	cv2.imshow('a', im)
# 	while count < 2:
# 		cv2.waitKey(10)
#
# 	data[im_path] = ['bar', 0, [[clicks[0][1], clicks[0][0], clicks[1][1] - clicks[0][1], clicks[1][0] - clicks[0][0]]]]
#
# 	(noun, noun_class, bbox) = data[im_path]
# 	for y, x, he, we in bbox:
# 		im = cv2.rectangle(im, (x, y), (x+we, y+he), (0, 0, 255), 2)
# 	cv2.imshow('a', im)
# 	cv2.waitKey(500)
#
# with open('boxes.pickle', 'wb') as handle:
# 	pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
