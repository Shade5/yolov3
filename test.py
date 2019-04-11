import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(
		dataloader,
		model=None,
		img_size=416,
		iou_thres=0.5,
		conf_thres=0.1,
		nms_thres=0.5,
		save_json=False
):
	device = next(model.parameters()).device  # get model device

	seen = 0
	model.eval()
	coco91class = coco80_to_coco91_class()
	print(('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
	loss, p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0., 0.
	jdict, stats, ap, ap_class = [], [], [], []
	for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Computing mAP')):
		targets = targets.to(device)
		imgs = imgs.to(device)

		# Run model
		inf_out, train_out = model(imgs)  # inference and training outputs

		# Build targets
		target_list = build_targets(model, targets)

		# Compute loss
		loss_i, _ = compute_loss(train_out, target_list)
		loss += loss_i.item()

		# Run NMS
		output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

		# Statistics per image
		for si, pred in enumerate(output):
			labels = targets[targets[:, 0] == si, 1:]
			correct, detected = [], []
			tcls = torch.Tensor()
			seen += 1

			if pred is None:
				if len(labels):
					tcls = labels[:, 0].cpu()  # target classes
					stats.append((correct, torch.Tensor(), torch.Tensor(), tcls))
				continue

			# Append to pycocotools JSON dictionary
			if save_json:
				# [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
				image_id = int(Path(paths[si]).stem.split('_')[-1])
				box = pred[:, :4].clone()  # xyxy
				scale_coords(img_size, box, shapes[si])  # to original shape
				box = xyxy2xywh(box)  # xywh
				box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
				for di, d in enumerate(pred):
					jdict.append({
						'image_id': image_id,
						'category_id': coco91class[int(d[6])],
						'bbox': [float3(x) for x in box[di]],
						'score': float(d[4])
					})

			if len(labels):
				# Extract target boxes as (x1, y1, x2, y2)
				tbox = xywh2xyxy(labels[:, 1:5]) * img_size  # target boxes
				tcls = labels[:, 0]  # target classes

				for *pbox, pconf, pcls_conf, pcls in pred:
					if pcls not in tcls:
						correct.append(0)
						continue

					# Best iou, index between pred and targets
					iou, bi = bbox_iou(pbox, tbox).max(0)

					# If iou > threshold and class is correct mark as correct
					if iou > iou_thres and bi not in detected:
						correct.append(1)
						detected.append(bi)
					else:
						correct.append(0)
			else:
				# If no labels add number of detections as incorrect
				correct.extend([0] * len(pred))

			# Append Statistics (correct, conf, pcls, tcls)
			stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls.cpu()))

	# Compute statistics
	stats_np = [np.concatenate(x, 0) for x in list(zip(*stats))]
	nt = np.bincount(stats_np[3].astype(np.int64), minlength=nc)  # number of targets per class
	if len(stats_np):
		p, r, ap, f1, ap_class = ap_per_class(*stats_np)
		mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

	# Print results
	pf = '%20s' + '%10.3g' * 6  # print format
	print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1), end='\n\n')

	# Return results
	return mp, mr, map, mf1, loss


if __name__ == '__main__':
	parser = argparse.ArgumentParser(prog='test.py')
	parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
	parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
	parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
	parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
	parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
	parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
	parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
	parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
	parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
	opt = parser.parse_args()
	print(opt, end='\n\n')

	with torch.no_grad():
		mAP = test(
			opt.cfg,
			opt.data_cfg,
			opt.weights,
			opt.batch_size,
			opt.img_size,
			opt.iou_thres,
			opt.conf_thres,
			opt.nms_thres,
			opt.save_json
		)
