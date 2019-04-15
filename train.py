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

    if resume:  # Load previously saved model
        if transfer:  # Transfer learning
            chkpt = torch.load(weights + 'yolov3.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)
            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from latest.pt
            chkpt = torch.load(latest, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

    # Set scheduler (reduce lr at epochs 218, 245, i.e. batches 400k, 450k)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[218, 245], gamma=0.1,
                                                     last_epoch=start_epoch - 1)

    # Dataset
    data_train = LoadEpic("data/object_detection_images/train", "data/boxes_small.pkl", img_size=img_size, augment=False)
    data_test = LoadEpic("data/object_detection_images/train", "data/boxes_small.pkl", img_size=img_size, augment=False)
    print("Total number of images:", len(data_train))

    validation_split = .2
    indices = list(range(len(data_train)))
    split = int(np.floor(validation_split * len(data_train)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # classes = load_classes("data/epic_veg.names")
    # for j in range(len(data_train)):
    #     im, l, _, _ = data_train[j]
    #     im = np.transpose(im.cpu().numpy(), (1, 2, 0))
    #     l = l.cpu().numpy()
    #     l[:, 2:] = l[:, 2:]*418
    #     l = l.astype(int)
    #     for i in range(l.shape[0]):
    #         _, cl, x, y, w, h = l[i]
    #         im = cv2.rectangle(im, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 255), 2)
    #         im = cv2.putText(im, classes[cl], (x - w//2, y - h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.imshow("a", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0)

    # Dataloader
    dataloader = DataLoader(data_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=data_train.collate_fn,
                            sampler=train_sampler)

    test_dataloader = DataLoader(data_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=data_train.collate_fn,
                            sampler=valid_sampler)

    # Start training
    t = time.time()
    model_info(model)
    nB = len(dataloader)
    n_burnin = min(round(nB / 5 + 1), 1000)  # burn-in batches
    os.remove('train_batch0.jpg') if os.path.exists('train_batch0.jpg') else None
    os.remove('test_batch0.jpg') if os.path.exists('test_batch0.jpg') else None
    n_iter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        print('Epoch', epoch)

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = defaultdict(float)  # mean loss
        for i, (imgs, targets, _, _) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(device)
            targets = targets.to(device)

            nt = len(targets)
            if nt == 0:  # if no targets continue
                continue

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Build targets
            target_list = build_targets(model, targets)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nB:
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            t = time.time()
            log(writer, mloss, n_iter)
            n_iter += 1

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                data_train.img_size = random.choice(range(10, 20)) * 32
                print('multi_scale img_size = %g' % data_train.img_size)

        # Calculate mAP
        with torch.no_grad():
            mp, mr, ap, mf1, tloss = test.test(test_dataloader, model=model, img_size=img_size)
            writer.add_scalar('test/precision', mp, n_iter)
            writer.add_scalar('test/recall', mr, n_iter)
            writer.add_scalar('test/map', ap, n_iter)
            writer.add_scalar('test/loss', tloss, n_iter)
        # Update best loss
        test_loss = tloss
        if test_loss < best_loss:
            best_loss = test_loss

        # Save training results
        save = True and not opt.nosave
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt


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
