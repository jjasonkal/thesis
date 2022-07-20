import json
import os.path as osp

import cv2
import numpy as np
import random

import torch
from PIL import Image
# from PIL.Image import Resampling
from matplotlib import pyplot as plt

from configs.global_vars import IMG_MEAN
# from data.augmentations import Compose, RandomCrop_gta
from utils.visualization import save_image
from data.augmentations import Compose, RandomCrop_gta, RandomCrop_city


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    sum_pixels = sum(overall_class_stats.values())
    freq = {
        k: v / sum_pixels
        for k, v in overall_class_stats.items()
    }
    # print(freq)
    values = [(1 - v) / temperature for v in freq.values()]
    prob = softmax(values)
    classes_probability = {
        list(freq.keys())[x]: prob[x]
        for x in range(len(values))
    }
    return classes_probability


def get_rcs_freq(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    sum_pixels = sum(overall_class_stats.values())
    freq = {
        k: v / sum_pixels
        for k, v in overall_class_stats.items()
    }
    # dict = {}
    # keys = list(freq.keys())
    # for x in classes:
    #     if x in keys:
    #         dict[x] = freq[x]
    # values = [(1 - v) / temperature for v in dict.values()]
    # prob = softmax(values)
    # classes_probability = {
    #     list(dict.keys())[x]: prob[x]
    #     for x in range(len(values))
    # }
    return freq


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_image_label_depth(data_root, classes_probability, augmentations, offset = None):
    random_class = random.choices(list(classes_probability.keys()), weights=classes_probability.values())[0]
    with open(osp.join(data_root, 'samples_with_class.json'), 'r') as of:
        samples_with_class = json.load(of)
    if random_class == 255:
        random_class = 0
    rnd = random.choice(samples_with_class[str(random_class)])
    path = rnd[0]
    pixels = rnd[1]
    image_path = path.replace('labels', 'images').replace('_labelTrainIds', '')  # TODO CHANGE images to new
    # print(image_path)
    label_path = path.replace('_labelTrainIds', '')
    depth_path = path.replace('labels', 'disparity').replace('_labelTrainIds', '')

    id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    img_size = (1280, 720)
    image = Image.open(image_path).convert('RGB')
    label = Image.open(label_path)
    depth = cv2.imread(depth_path, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256. + 1.
    name = image_path.split("/")[-1].split(".")[0]

    # resize
    image = image.resize(img_size, resample=Image.BICUBIC)
    label = label.resize(img_size, resample=Image.NEAREST)
    depth = cv2.resize(depth, img_size, interpolation=cv2.INTER_LINEAR)

    image = np.asarray(image, np.uint8)
    label = np.asarray(label, np.uint8)

    if offset == None:
        if augmentations is not None:
            image, label, depth = augmentations(image, label, depth)
        # print('here')
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
    else:
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        from random import randrange

        cropx = int(round(1280*offset/1024))
        cropy = randrange(720-512)
        image = image[cropy:cropy+512, 0 + cropx:512 + cropx, :]
        label = label[cropy:cropy+512, 0 + cropx:512 + cropx]
        depth = depth[cropy:cropy+512, 0 + cropx:512 + cropx]

    # image = np.asarray(image, np.float32)
    # label = np.asarray(label, np.float32)

    # PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
    # PIL_label = Image.fromarray(label.astype('uint8'))
    # plt.imshow(PIL_image)
    # plt.show()
    # plt.imshow(PIL_label)
    # plt.show()
    # plt.imshow(depth)
    # plt.show()

    # re-assign labels to match the format of Cityscapes
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
    for k, v in id_to_trainid.items():
        label_copy[label == k] = v

    size = image.shape
    image = image[:, :, ::-1]  # change to BGR
    image -= IMG_MEAN
    image = image.transpose((2, 0, 1))

    return image.copy(), label_copy.copy(), np.array(size), depth

def get_depth(file):
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)/100.
    depth = 655.36 / (depth + 0.01)
    return depth

def get_image_label_depth_synthia(data_root, classes_probability, augmentations, offset = None):
    random_class = random.choices(list(classes_probability.keys()), weights=classes_probability.values())[0]
    # print(random_class)
    with open(osp.join(data_root, 'samples_with_class.json'), 'r') as of:
        samples_with_class = json.load(of)
    if random_class == 255:
        random_class = 0
    rnd = random.choice(samples_with_class[str(random_class)])
    path = rnd[0]
    # pixels = rnd[1]
    # image_path = path.replace('labels', 'images').replace('_labelTrainIds', '')  # TODO CHANGE images to new
    # # print(image_path)
    # label_path = path.replace('_labelTrainIds', '')
    # depth_path = path.replace('labels', 'disparity').replace('_labelTrainIds', '')

    # PATH?? imgsize???
    img_size=(512, 512)
    arr = rnd[0].split("/")
    name = arr[len(arr)-1][:7] + '.png'

    image = Image.open(osp.join(data_root, "RGB/%s" % name)).convert('RGB')
    # label = Image.open(osp.join(data_root, "GT/LABELS/%s" % name))
    label = Image.open(osp.join(data_root, "synthia_mapped_to_cityscapes/%s" % name))
    depth = get_depth(osp.join(data_root, "Depth/Depth/%s" % name))

    # resize
    image = image.resize(img_size, Image.BICUBIC)
    label = label.resize(img_size, Image.NEAREST)
    depth = cv2.resize(depth, img_size, interpolation=cv2.INTER_NEAREST)

    image = np.asarray(image, np.uint8)
    label = np.asarray(label, np.uint8)

    # if augmentations is not None:
    #     image, label, depth = augmentations(image, label, depth)

    image = np.asarray(image, np.float32)
    label = np.asarray(label, np.float32)

    # re-assign labels to match the format of Cityscapes
    label_copy = 255 * np.ones(label.shape, dtype=np.float32)

    id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                     15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                     8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}

    for k, v in id_to_trainid.items():
        label_copy[label == k] = v

    size = image.shape
    image = image[:, :, ::-1]  # change to BGR
    image -= IMG_MEAN
    image = image.transpose((2, 0, 1))

    return image.copy(), label_copy.copy(), np.array(size), depth  # .unsqueeze(1)#name


path = "/mnt/genesis/kaltsikis/data/gta"
classes_probability = get_rcs_class_probs(path, 0.01)
# mix_probability = get_rcs_freq(path, 0.2)
print(classes_probability)
# print(mix_probability)
# data_aug = Compose([RandomCrop_gta((512, 512))])
# image1, label1, _, depth1 = get_image_label_depth(path, classes_probability, data_aug)
# image2, label2, _, depth2 = get_image_label_depth(path, classes_probability, data_aug)
# images = [image1,image2]
# labels = [label1,label2]
# depth = [depth1,depth2]

#
# temperature = 0.05
# classes_freq = get_rcs_freq(path, 0.2)
# classes = torch.Tensor([1,255,5,3,0,2])
# # print(classes)
# nclasses = classes.shape[0]
# # print(nclasses)
# list_classes = classes.tolist()
# list_classes = [int(x) for x in list_classes]
# dict = {}
# keys = list(classes_freq.keys())
# # print(classes_freq)
# for x in list_classes:
#     if x != 255 and x in keys:
#         dict[x] = classes_freq[x]
# # print(dict)
# values = [(1 - v) / temperature for v in dict.values()]
# prob = softmax(values)
# classes_probability = {
#     list(dict.keys())[x]: prob[x]
#     for x in range(len(values))
# }
# print(classes_probability)
# # print(list_classes)
# # print(len(list_classes))
# # print(classes_probability)
# probs = list(classes_probability.values())
# new_classes = torch.as_tensor(list(classes_probability.keys())).cuda()
# nnclasses = new_classes.shape[0]
# print(new_classes)
# # print(new_classes.shape[0])
# # print(probs)
# # print(len(probs))
# # print(classes)
# # only nan not equal to self
# # probs = [0.0 if x!=x else x for x in probs]
# print(probs)
# print(len(probs))
# if nnclasses > 0:
#     new_classes = (new_classes[torch.Tensor(np.random.choice(nnclasses, int((nclasses+nclasses%2)/2),p=probs,replace=False)).long()]).cuda()
#     print(new_classes)

# path = "/mnt/genesis/kaltsikis/data/RAND_CITYSCAPES"
# classes_probability_1 = get_rcs_class_probs(path, 0.2)
# classes_freq = get_rcs_freq(path, 0.01)
# print(classes_freq)
# print(classes_probability_1)
#
# # data_aug = Compose([RandomCrop_city((512, 512))])
# data_aug = None
# for x in range(10):
#     image1, label1, _, depth1 = get_image_label_depth_synthia(path, classes_probability_1, data_aug)
# save_image('/home/kaltsikis/corda/synthia_test', torch.from_numpy(image1), 'synthia_image', '')
# save_image('/home/kaltsikis/corda/synthia_test', torch.from_numpy(label1), 'synthia_label', '')
# save_image('/home/kaltsikis/corda/synthia_test', torch.from_numpy(depth1), 'synthia_depth', '')



class_names = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic_light",
    "traffic_sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
