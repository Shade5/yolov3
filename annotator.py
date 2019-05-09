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


# Make sure the path only has images
data_path = "/home/a/Downloads/bar/far"
data_path_annotated = "/home/a/Downloads/bar/indoor_annotated"

offset = 0

try:
	with open(data_path_annotated + '/boxes.pkl', 'rb') as handle:
		data = pickle.load(handle)
	print("Found dataset")
	offset = len(data.keys())
except:
	print("New dataset")
	data = {}
if not os.path.exists(data_path_annotated):
	os.makedirs(data_path_annotated)

window_name = 'Click on the top left and bottom right points'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, onMouse)

for i, im_path in enumerate(tqdm(glob(data_path + "/*"))):
	i += offset
	while True:
		count = 0
		clicks = []
		im = cv2.imread(im_path)
		cv2.imwrite(data_path_annotated + "/" + str(i) + ".png", im)
		im = cv2.resize(im, (416, 416))
		cv2.imshow(window_name, im)
		while count < 2:
			cv2.waitKey(10)

		data[str(i) + ".png"] = [
			['barrel', 1, [[clicks[0][1], clicks[0][0], clicks[1][1] - clicks[0][1], clicks[1][0] - clicks[0][0]]]]]

		for (noun, noun_class, bbox) in data[str(i) + ".png"]:
			for y, x, he, we in bbox:
				im = cv2.rectangle(im, (x, y), (x + we, y + he), (0, 0, 255), 2)

		cv2.putText(im, 'Press Enter to confirm, any other key to retry', (0, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.imshow(window_name, im)
		k = cv2.waitKey(0)
		if k == 13:  # Enter key to stop
			break
		else:  # normally -1 returned,so don't print it
			continue

with open(data_path_annotated + '/boxes.pkl', 'wb') as handle:
	pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

cv2.destroyAllWindows()

print("All annotations")

with open(data_path_annotated + '/boxes.pkl', 'rb') as handle:
	data_annotated = pickle.load(handle)

for i, im_path in enumerate(glob(data_path_annotated + "/*.png")):
	im = cv2.resize(cv2.imread(im_path), (416, 416))
	entries = data_annotated[os.path.basename(im_path)]
	for (noun, noun_class, bbox) in entries:
		for y, x, he, we in bbox:
			im = cv2.rectangle(im, (x, y), (x+we, y+he), (0, 0, 255), 2)
	cv2.imshow('Annotations', im)
	cv2.waitKey(10)
