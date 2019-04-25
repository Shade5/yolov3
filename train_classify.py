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
	writer_train = SummaryWriter('./tbx/' + "YOLO_CLASS" + str(datetime.now())[:-7] + "/train")

	data_train = VOC("./scraped100", split='trainval', transform=data_transforms['train'])
	data_test = VOC("./scraped100", split='test', transform=data_transforms['val'])

	validation_split = .2

	indices = list(range(len(data_train)))
	split = int(np.floor(validation_split * len(data_train)))
	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	dataloader_train = DataLoader(data_train, batch_size=32, sampler=train_sampler, num_workers=4)
	dataloader_test = DataLoader(data_test, batch_size=32, sampler=valid_sampler, num_workers=4)

	weights = 'weights' + os.sep
	latest = weights + 'latest.pt'
	best = weights + 'best.pt'
	device = torch_utils.select_device()
	run_name = "fixed" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	writer = SummaryWriter("./tbx/" + run_name)
	print("Run name:", run_name)

	if multi_scale:
		img_size = 608  # initiate with maximum multi_scale size
		num_workers = 0  # bug https://github.com/ultralytics/yolov3/issues/174
	else:
		torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

	# Configure run
	train_path = parse_data_cfg(data_cfg)['train']

	# Initialize model
	model = Darknet(cfg, img_size).to(device)

	# Optimizer
	lr0 = 0.0001  # initial learning rate
	optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)

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
	optimizer = optim.Adam(model.parameters())

	global_step = 0

	for e in range(20):
		model.train()
		for batch, (inputs, labels, weights) in enumerate(tqdm(dataloader_train)):
			inputs = inputs.to(device)
			labels = labels.to(device)
			weights = weights.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward
			# track history if only in train
			with torch.set_grad_enabled(True):
				outputs = torch.sigmoid(model(inputs, classify=True))
				loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()
			writer_train.add_scalar('train/Loss', loss, global_step)

			global_step += 1

		print(loss)

		if e % 2 == 0:
			torch.save(model.state_dict(), "./pytorch_models/" + str(e))
			AP, mAP, acc = eval_dataset_map(model, dataloader_test, device)
			writer_train.add_scalar('val/mAP', mAP, global_step)
			writer_train.add_scalar('val/acc', acc, global_step)
			print("Test mAP, acc", mAP, acc)
			AP, mAP, acc = eval_dataset_map(model, dataloader_train, device)
			writer_train.add_scalar('train/mAP', mAP, global_step)
			writer_train.add_scalar('train/acc', acc, global_step)
			print("Train mAP, acc", mAP, acc)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
	parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
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
