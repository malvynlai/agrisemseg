from __future__ import division

import os
import torch
import torchvision.transforms as st
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import cv2
from model.res_class_model import ResNet50Classifier
from data.loader import load_test_images, load_ids, get_test_list

output_path = os.path.join('./submission', 'results_r101')

def main():
    models = []

    model = load_model(name='MSCG-Rx101', classes=9, node_size=(32, 32))

    checkpoint = torch.load('ckpt/R101_baseline/epoch_20_loss_1.09793_acc_0.78908_acc-cls_0.61996_mean-iu_0.47694_fwavacc_0.65960_f1_0.63160_lr_0.0000946918.pth')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    models.append(model)

    test_files = get_test_list(bands=['NIR', 'RGB'])
    test_time_augmentation(models, stride=600, batch_size=1, norm=False, window_size=(512, 512), labels=land_classes, test_set=test_files, all=True)

def predict_with_fusion(models, image, scales, batch_size=1, num_class=7, wsize=(512, 512)):
    pred_all = np.zeros(image.shape[:2] + (num_class,))
    for scale_rate in scales:
        img = image.copy()
        pred = np.zeros(img.shape[:2] + (num_class,))
        stride = img.shape[1]
        window_size = img.shape[:2]
        for i, coords in enumerate(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            imgs_flip = [patch[:, ::-1, :] for patch in image_patches]
            imgs_mirror = [patch[:, :, ::-1] for patch in image_patches]

            image_patches = np.concatenate((image_patches, imgs_flip, imgs_mirror), axis=0)
            image_patches = np.asarray(image_patches)
            image_patches = torch.from_numpy(image_patches).cuda()

            for model in models:
                outs = model(image_patches)
                outs = outs.data.cpu().numpy()

                b, _, _, _ = outs.shape

                for out, (x, y, w, h) in zip(outs[0:b // 3, :, :, :], coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                for out, (x, y, w, h) in zip(outs[b // 3:2 * b // 3, :, :, :], coords):
                    out = out[:, ::-1, :]
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                for out, (x, y, w, h) in zip(outs[2 * b // 3: b, :, :, :], coords):
                    out = out[:, :, ::-1]
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

                del (outs)

    return pred_all

def test_time_augmentation(models, all=False, labels=None, norm=False, test_set=None, stride=600, batch_size=5, window_size=(512, 512)):
    test_files = load_test_images(test_set)
    idlist = load_ids(test_set)

    all_preds = []
    num_class = len(labels)
    ids = []

    total_ids = 0

    for k in test_set.keys():
        total_ids += len(test_set[k])

    for img, id in tqdm(zip(test_files, idlist), total=total_ids, leave=False):
        img = np.asarray(img, dtype='float32')
        img = st.ToTensor()(img)
        img = img / 255.0
        if norm:
        img = img.cpu().numpy().transpose((1, 2, 0))

        with torch.no_grad():
            pred = np.argmax(pred, axis=-1)

        for key in ['boundaries', 'masks']:
            pred = pred * np.array(cv2.imread(os.path.join('/home/songyao/workspace/data/supervised/Agriculture-Vision-2021/test', key, id + '.png'), -1) / 255, dtype=int)
        filename = './{}.png'.format(id)
        cv2.imwrite(os.path.join(output_path, filename), pred)

        all_preds.append(pred)
        ids.append(id)

    if all:
        return all_preds, ids
    else:
        return all_preds

def metrics(predictions, gts, label_values=None):
    cm = confusion_matrix(gts, predictions, range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    print("---")

    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print("---")

    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except BaseException:
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")

    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    return accuracy, cm

def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def count_sliding_window(top, step=10, window_size=(20, 20)):
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple([next(it, None) for _ in range(n)])
        chunk = tuple(filter(lambda x: x is not None, chunk))
        if not chunk:
            return
        yield chunk


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()
