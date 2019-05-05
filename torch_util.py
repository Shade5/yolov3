from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore")

CLASS_NAMES = ['grapes', 'pineapples', 'kiwi', 'cucumber', 'egg', 'lettuce', 'olives', 'onion', 'tomato', "Beans", "Mushroom", "Artichokes", "Pears", "Cabbage", "pasta", "broccoli", "apples", "Lemons", "Corn", "chicken", "potatoes", "carrots", "avocado", "bacon"]


class VOC(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.image_names = np.loadtxt(data_dir + "/ing_list/" + CLASS_NAMES[0] + ".txt").astype(np.int)[:, 0]
        self.image_dir = data_dir + "/images/"
        self.labels = np.zeros((self.image_names.shape[0], len(CLASS_NAMES)), dtype=np.float32)
        self.weights = np.zeros((self.image_names.shape[0], len(CLASS_NAMES)), dtype=np.float32)
        self.transform = transform
        for i, c in enumerate(CLASS_NAMES):
            self.labels[:, i] = (np.loadtxt(data_dir + "/ing_list/" + c + ".txt") > -0.5).astype(np.int)[:, 1]
            self.weights[:, i] = (np.abs(np.loadtxt(data_dir + "/ing_list/" + c + ".txt")) > 0.5).astype(np.int)[:, 1]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + str(self.image_names[idx]) + ".jpg").convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], self.weights[idx]


def compute_ap(gt, pred, valid, average=None):
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset, device):
    model.eval()
    for batch, (images, labels, weights) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        output = model(images, classify=True)
        if batch == 0:
            gt = labels.cpu().detach().numpy()
            pred = output.cpu().detach().numpy()
            valid = weights.cpu().detach().numpy()
        else:
            gt = np.append(gt, labels.cpu().detach().numpy(), axis=0)
            pred = np.append(pred, output.cpu().detach().numpy(), axis=0)
            valid = np.append(valid, weights.cpu().detach().numpy(), axis=0)

    AP = compute_ap(gt, pred, valid)
    acc = np.sum(np.argmax(pred, axis=1) == np.argmax(gt, axis=1))/pred.shape[0]
    return AP, np.mean(AP), acc
