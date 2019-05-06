import argparse
import time

import torch.distributed as dist
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch_util import VOC, eval_dataset_map
from torch import optim

def log(writer, mloss, n_iter):
	writer.add_scalar('loss/x+y', mloss['xy'], n_iter)
	writer.add_scalar('loss/w+h', mloss['wh'], n_iter)
	writer.add_scalar('loss/conf', mloss['conf'], n_iter)
	writer.add_scalar('loss/cls', mloss['cls'], n_iter)
	writer.add_scalar('loss/total', mloss['total'], n_iter)


def train_step_class(model, data, optimizer, criterion, device):
	(inputs, labels, weights) = data
	inputs = inputs.to(device)
	labels = labels.to(device)

	optimizer.zero_grad()
	outputs = torch.sigmoid(model(inputs, classify=True))
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()

	return loss


def train_step_detect(model, data, optimizer, device):
	(imgs, targets, _, _) = data
	imgs = imgs.to(device)
	targets = targets.to(device)

	nt = len(targets)
	if nt == 0:  # if no targets continue
		return 0

	optimizer.zero_grad()
	pred = model(imgs)
	target_list = build_targets(model, targets)
	loss, loss_dict = compute_loss(pred, target_list)
	loss.backward()
	optimizer.step()

	return loss


def train(
		cfg,
		data_cfg,
		img_size=416,
		resume=False,
		epochs=273,  # 500200 batches at bs 64, dataset length 117263
		batch_size=16,
		accumulate=1,
		multi_scale=False,
		freeze_backbone=False,
		num_workers=4,
		transfer=False  # Transfer learning (train only YOLO layers)

):
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(416),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])
		]),
		'val': transforms.Compose([
			transforms.Resize(512),
			transforms.CenterCrop(416),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [1, 1, 1])
		]),
	}

	# Classifier Dataloader
	data_train_class = VOC("./scraped100", split='trainval', transform=data_transforms['train'])
	data_test_class = VOC("./scraped100", split='test', transform=data_transforms['val'])
	print("Classifier images:", len(data_train_class), "Batch size:", batch_size)

	validation_split = .2

	indices = list(range(len(data_train_class)))
	split = int(np.floor(validation_split * len(data_train_class)))
	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	dataloader_train_class = DataLoader(data_train_class, batch_size=batch_size, sampler=train_sampler, num_workers=4)
	dataloader_test_class = DataLoader(data_test_class, batch_size=batch_size, sampler=valid_sampler, num_workers=4)


	# Classifier Detector
	data_train_detect = LoadEpic("data/object_detection_images/train", "data/boxes_common.pkl", img_size=img_size, augment=False)
	data_test_detect = LoadEpic("data/object_detection_images/train", "data/boxes_common.pkl", img_size=img_size, augment=False)
	print("Detector images:", len(data_train_detect), "Batch size:", batch_size * len(data_train_detect) / len(data_train_class))

	validation_split = .2

	indices = list(range(len(data_train_detect)))
	split = int(np.floor(validation_split * len(data_train_detect)))
	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	dataloader_train_detect = DataLoader(data_train_detect, batch_size=int(batch_size * len(data_train_detect) / len(data_train_class)), sampler=train_sampler, collate_fn=data_train_detect.collate_fn)
	dataloader_test_detect = DataLoader(data_test_detect, batch_size=int(batch_size * len(data_train_detect) / len(data_train_class)), sampler=valid_sampler, collate_fn=data_test_detect.collate_fn)

	weights = 'weights' + os.sep
	latest = weights + 'latest.pt'
	best = weights + 'best.pt'
	device = torch_utils.select_device()
	run_name = "hydra_" + input("Enter run name") + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	writer_train = SummaryWriter("./tbx/" + run_name)
	print("Run name:", run_name)

	torch.backends.cudnn.benchmark = True

	# Configure run
	train_path = parse_data_cfg(data_cfg)['train']

	# Initialize model
	model = Darknet(cfg, img_size)

	model.load_state_dict(torch.load("weights/food36_classify.pt", map_location=device))

	model.to(device)

	cutoff = -1  # backbone reaches to cutoff layer
	start_epoch = 0
	best_loss = float('inf')
	yl = get_yolo_layers(model)  # yolo layers
	nf = int(model.module_defs[yl[0] - 1]['filters'])  # yolo layer size (i.e. 255)

	chkpt = torch.load(weights + 'yolov3.pt', map_location=device)
	model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255}, strict=False)
	for p in model.parameters():
		p.requires_grad = True if p.shape[0] == nf else False

	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	global_step = 0

	for e in range(200):
		model.train()
		for batch, data in enumerate(tqdm(zip(dataloader_train_class, dataloader_train_detect))):
			loss = train_step_class(model, data[0], optimizer, criterion, device)
			writer_train.add_scalar('train/Classifier_Loss', loss, global_step)
			loss = train_step_detect(model, data[1], optimizer, device)
			writer_train.add_scalar('train/Detector_Loss', loss, global_step)

			global_step += 1

		if e % 2 == 0:
			model.eval()
			torch.save(model.state_dict(), "./pytorch_models/" + str(e))
			AP, mAP, acc = eval_dataset_map(model, dataloader_test_class, device)
			writer_train.add_scalar('val/Classifier_mAP', mAP, global_step)
			writer_train.add_scalar('val/Classifier_acc', acc, global_step)
			AP, mAP, acc = eval_dataset_map(model, dataloader_test_class, device)
			writer_train.add_scalar('train/Classifier_mAP', mAP, global_step)
			writer_train.add_scalar('train/Classifier_acc', acc, global_step)
			with torch.no_grad():
				mp, mr, ap, mf1, tloss = test.test(dataloader_test_detect, model=model, img_size=img_size)
			writer_train.add_scalar('val/Detector_precision', mp, global_step)
			writer_train.add_scalar('val/Detector_recall', mr, global_step)
			writer_train.add_scalar('val/Detector_map', ap, global_step)
			writer_train.add_scalar('val/Detector_loss', tloss, global_step)
			with torch.no_grad():
				mp, mr, ap, mf1, tloss = test.test(dataloader_train_detect, model=model, img_size=img_size)
			writer_train.add_scalar('train/Detector_precision', mp, global_step)
			writer_train.add_scalar('train/Detector_recall', mr, global_step)
			writer_train.add_scalar('train/Detector_map', ap, global_step)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
	parser.add_argument('--batch-size', type=int, default=10, help='size of each image batch')
	parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
	parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
	parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
	parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
	parser.add_argument('--img-size', type=int, default=416, help='pixels')
	parser.add_argument('--resume', action='store_true', help='resume training flag')
	parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
	parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
	parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
	parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
	parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
	parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
	parser.add_argument('--nosave', action='store_true', help='do not save training results')
	opt = parser.parse_args()
	print(opt, end='\n\n')

	init_seeds()

	train(
		opt.cfg,
		opt.data_cfg,
		img_size=opt.img_size,
		resume=True,
		transfer=True,
		epochs=opt.epochs,
		batch_size=opt.batch_size,
		accumulate=opt.accumulate,
		multi_scale=opt.multi_scale,
		num_workers=opt.num_workers
	)
