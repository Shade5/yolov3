import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        data_cfg,
        weights,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    model = Darknet(cfg, img_size)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadPickle("/home/a/Downloads/bar/indoor_annotated", "/home/a/Downloads/bar/indoor_annotated/boxes.pkl", img_size=img_size, augment=False)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    if webcam:
        for i, (img_path, img, im0, _) in enumerate(dataloader):
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            pred, _ = model(img)
            det = non_max_suppression(pred, conf_thres, nms_thres)[0]

            if det is not None and len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls_conf, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            cv2.imshow(weights, im0)
            cv2.waitKey(1)

    else:
        for i in range(len(dataloader)):
            (img, targets, _, _) = dataloader[np.random.randint(0, len(dataloader))]

            # Get detections
            im0 = cv2.cvtColor(np.transpose(img.numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            img = img.unsqueeze(0).to(device)
            pred, _ = model(img)
            det = non_max_suppression(pred, conf_thres, nms_thres)[0]

            if det is not None and len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls_conf, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            cv2.imshow("a", im0)
            cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/backup130.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            webcam=True
        )
