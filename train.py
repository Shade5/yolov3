import argparse
import time

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test  # Import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler

# Hyperparameters
hyp = {'k': 10.39,  # loss multiple
       'xy': 0.1367,  # xy loss fraction
       'wh': 0.01057,  # wh loss fraction
       'cls': 0.01181,  # cls loss fraction
       'conf': 0.8409,  # conf loss fraction
       'iou_t': 0.1287,  # iou target-anchor training threshold
       'lr0': 0.001028,  # initial learning rate
       'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9127,  # SGD momentum
       'weight_decay': 0.0004841,  # optimizer weight decay
       }


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=True,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        freeze_backbone=False,
        transfer=True  # Transfer learning (train only YOLO layers)
):
    init_seeds()
    weights = 'weights' + os.sep
    latest = weights + 'latest.pt'
    device = torch_utils.select_device()
    run_name = input("Enter run name: ") + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter("./tbx/" + run_name)
    print("Run name:", run_name)

    torch.backends.cudnn.benchmark = True

    # Configure run
    data_dict = parse_data_cfg(data_cfg)

    # Initialize model
    model = Darknet(cfg, img_size).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
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

    lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

    # Dataset
    data_train = LoadEpic("data/object_detection_images/train",
                          "data/boxes_common.pkl",
                          img_size=img_size,
                          augment=False)
    data_val = LoadEpic("data/object_detection_images/train",
                         "data/boxes_common.pkl",
                         img_size=img_size,
                         augment=False)
    print("Total number of images:", len(data_train))

    validation_split = .2
    indices = list(range(len(data_train)))
    split = int(np.floor(validation_split * len(data_train)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # for j in range(len(data_train)):
    #     im, l, _, _ = data_train[j]
    #     im = np.transpose(im.cpu().numpy(), (1, 2, 0))
    #     l = l.cpu().numpy()
    #     l[:, 2:] = l[:, 2:]*418
    #     l = l.astype(int)
    #     for i in range(l.shape[0]):
    #         _, cl, x, y, w, h = l[i]
    #         im = cv2.rectangle(im, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 255), 2)
    #         # im = cv2.putText(im, classes[cl], (x - w//2, y - h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.imshow("a", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0)

    # Dataloader
    dataloader = DataLoader(data_train,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            collate_fn=data_train.collate_fn,
                            sampler=train_sampler)

    test_dataloader = DataLoader(data_val,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            collate_fn=data_train.collate_fn,
                            sampler=valid_sampler)

    # Start training
    t, t0 = time.time(), time.time()
    model.hyp = hyp  # attach hyperparameters to model
    model_info(model, report='full')
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    n_iter = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets, _, _) in enumerate(tqdm(dataloader, desc='Epoch ' + str(epoch))):
            imgs = imgs.to(device)
            targets = targets.to(device)
            nt = len(targets)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            n_iter += 1

            # Update running mean of tracked metrics
            mloss = (mloss * i + loss_items) / (i + 1)
            writer.add_scalar('train/loss_xy', mloss[0], n_iter)
            writer.add_scalar('train/loss_wh', mloss[1], n_iter)
            writer.add_scalar('train/loss_conf', mloss[2], n_iter)
            writer.add_scalar('train/loss_cls', mloss[3], n_iter)
            writer.add_scalar('train/loss', mloss[4], n_iter)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if epoch % 10 == 0 and epoch > 0:
            with torch.no_grad():
                model.eval()
                mp, mr, ap, mf1, tloss = test.test(test_dataloader, model=model, conf_thres=0.1)
                writer.add_scalar('val/precision', mp, n_iter)
                writer.add_scalar('val/recall', mr, n_iter)
                writer.add_scalar('val/map', ap, n_iter)
                writer.add_scalar('val/loss', tloss, n_iter)

                mp, mr, ap, mf1, tloss = test.test(dataloader, model=model, conf_thres=0.1)
                writer.add_scalar('train/precision', mp, n_iter)
                writer.add_scalar('train/recall', mr, n_iter)
                writer.add_scalar('train/map', ap, n_iter)
            model.train()

            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    dt = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch, dt))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()
    print(opt)

    # Train
    results = train(
        opt.cfg,
        opt.data_cfg,
        img_size=opt.img_size,
        resume=True,
        transfer=True,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
    )
