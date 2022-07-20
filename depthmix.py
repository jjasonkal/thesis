import torch
import numpy as np
import random
# from PIL import Image
# from matplotlib import pyplot as plt

from configs.global_vars import IMG_MEAN
from data.augmentations import Compose, RandomCrop_gta
from utils import transformmasks
from utils.visualization import save_image

from uda_dataset import get_rcs_class_probs, get_image_label_depth
from utils.visualization import DeNormalize
from data import cityscapesLoader

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

def class_mix(source_image, source_label, source_depth, target_image, target_depth):
    # mask = MixMask1.cpu().detach().numpy()
    mask = 0
    for k in selected:
        mask = np.where(source_label == k, 1, mask)

    # strong_parameters = {"Mix": MixMask1}
    # strong_parameters["flip"] = 0
    # strong_parameters["ColorJitter"] = 0
    # strong_parameters["GaussianBlur"] = 0
    augmented = np.where(mask == 1, source_image, target_image)
    # # inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat(torch.from_numpy(images[0]).unsqueeze(0), torch.from_numpy(images[1]).unsqueeze(0)))
    save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(augmented), 'augmented_ClassMix', '')


def depth_mix_half_classes(source_image, source_label, source_depth, target_image, target_depth):
    min_depth = 0.1
    max_depth = 0.4

    # classes = torch.unique(torch.from_numpy(source_label).cuda().long())
    # nclasses = classes.shape[0]
    # classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).cuda()
    # MixMask = transformmasks.generate_class_mask(torch.from_numpy(source_label).cuda().long(), classes).unsqueeze(0).cuda()
    # np_arr = MixMask.cpu().detach().numpy()
    mask = np.where(random.uniform(0, 1) * (max_depth - min_depth) + min_depth + source_depth >= target_depth, np_arr,
                    0)

    MixMask = mask
    strong_parameters = {"Mix": MixMask}
    strong_parameters["flip"] = 0
    strong_parameters["ColorJitter"] = 0
    strong_parameters["GaussianBlur"] = 0
    augmented = np.where(mask == 1, source_image, target_image)
    # inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat(torch.from_numpy(images[0]).unsqueeze(0), torch.from_numpy(images[1]).unsqueeze(0)))
    save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(augmented), 'augmented_DepthMix', '')


def depth_mix_half_classes_min_depth(source_image, source_label, source_depth, target_image, target_depth, MixMask):
    min_depth = 1
    max_depth = 2

    # classes = torch.unique(torch.from_numpy(source_label).cuda().long())
    # nclasses = classes.shape[0]
    # classes = (
    # classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).cuda()
    # MixMask = transformmasks.generate_class_mask(torch.from_numpy(source_label).cuda().long(), classes).unsqueeze(
    #     0).cuda()
    # np_arr = MixMask.cpu().detach().numpy()
    # selected_classes = classes.cpu().detach().numpy()
    mask = MixMask.cpu().detach().numpy()
    mask = 0
    rows = source_label.shape[0]
    cols = source_label.shape[1]
    for k in selected_classes:
        d1 = []
        d2 = []
        for x in range(0, cols - 1):
            for y in range(0, rows - 1):
                if source_label[x, y] == k:
                    d1.append(source_depth[x, y])
                    d2.append(target_depth[x, y])
        mask = np.where(min(d1) < min(d2) and source_label == k, 1, mask)
    MixMask = mask
    strong_parameters = {"Mix": MixMask}
    strong_parameters["flip"] = 0
    strong_parameters["ColorJitter"] = 0
    strong_parameters["GaussianBlur"] = 0
    augmented = np.where(mask == 1, source_image, target_image)
    # # inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat(torch.from_numpy(images[0]).unsqueeze(0), torch.from_numpy(images[1]).unsqueeze(0)))
    save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(augmented), 'augmented_DepthMix_min', '')


def median(l):
    half = len(l) // 2
    l.sort()
    return l[half]


def depth_mix_half_classes_median_depth(source_image, source_label, source_depth, target_image, target_depth, MixMask1):
    min_depth = 1
    max_depth = 2

    # classes = torch.unique(torch.from_numpy(source_label).cuda().long())
    # nclasses = classes.shape[0]
    # classes = (
    # classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).cuda()
    # MixMask = transformmasks.generate_class_mask(torch.from_numpy(source_label).cuda().long(), classes).unsqueeze(
    #     0).cuda()
    # np_arr = MixMask.cpu().detach().numpy()
    # selected_classes = classes.cpu().detach().numpy()
    mask = MixMask1.cpu().detach().numpy()
    mask = 0
    rows = source_label.shape[0]
    cols = source_label.shape[1]
    for k in selected_classes:
        d1 = []
        d2 = []
        for x in range(0, cols - 1):
            for y in range(0, rows - 1):
                if source_label[x, y] == k:
                    d1.append(source_depth[x, y])
                    d2.append(target_depth[x, y])
        mask = np.where(median(d1) < median(d2) and source_label == k, 1, mask)
    MixMask = mask
    strong_parameters = {"Mix": MixMask}
    strong_parameters["flip"] = 0
    strong_parameters["ColorJitter"] = 0
    strong_parameters["GaussianBlur"] = 0
    augmented = np.where(mask == 1, source_image, target_image)
    # # inputs_u_s0, _ = strongTransform(strong_parameters, data = torch.cat(torch.from_numpy(images[0]).unsqueeze(0), torch.from_numpy(images[1]).unsqueeze(0)))
    save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(augmented), 'augmented_DepthMix_median', '')


path = "/mnt/genesis/kaltsikis/data/gta"  # TODO NEW
classes_probability = get_rcs_class_probs(path, 0.01)
# print(classes_probability)
data_aug = Compose([RandomCrop_gta((512, 512))])
# image1, source_label, _, depth1 = get_image_label_depth(path, classes_probability, data_aug)
import os, random

city_path = random.choice(
    os.listdir("/mnt/genesis/kaltsikis/data/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/"))
image_path = random.choice(
    os.listdir("/mnt/genesis/kaltsikis/data/CityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/" + city_path))
print(city_path + "/" + image_path)
target_image, target_label, _, target_depth = cityscapesLoader.getitem(city_path + "/" + image_path)
# MATCHING GEOMETRY SAMPLING
min_difference = 999999
for x in range(3):
    image, label, _, depth = get_image_label_depth(path, classes_probability, data_aug)
    difference = np.sum(np.abs(np.subtract(np.log(1. / target_depth), np.log(1. / depth))))
    if difference < min_difference:
        min_difference = difference
        image1, source_label, _, depth1 = image, label, _, depth
print(min_difference)

save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(target_image), 'target_image', '')
save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(image1), 'source_image', '')
save_image('/home/kaltsikis/corda/depthmix_samples', torch.from_numpy(source_label), 'source_label', '')


# classes = torch.unique(torch.from_numpy(source_label).cuda().long())
# nclasses = classes.shape[0]
classes = np.unique(source_label)

# RareClassMix
sorted_dict = {k: v for k, v in sorted(classes_probability.items(), key=lambda item: item[1])}
prefer = list(sorted_dict.keys())[::-1]
print(prefer)

# classes = torch.Tensor([0, 1, 10, 18, 6, 0, 13])
list_classes = classes.tolist()
print(list_classes)
print([class_names[int(x)] for x in list_classes if int(x)!=255])
from math import ceil
selected = []
for x in prefer:
    if x in list_classes:
        selected.append(x)
        if len(selected) >= ceil(len(list_classes) / 2):
            break

print(selected)
print([class_names[int(x)] for x in selected if int(x)!=255])

# print(classes)
classes = torch.Tensor(np.asarray(selected))
# classes = classes.int()
# print(classes)


# # Random half classes
# classes = (
# classes[torch.Tensor(np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)).long()]).cuda()
# selected_classes = classes.cpu().detach().numpy()

# # Least Geometric difference classes
# from math import log
# all_classes = classes.cpu().detach().numpy()
# print(all_classes)
# dict = {}
# rows = source_label.shape[0]
# cols = source_label.shape[1]
# for k in all_classes:
#     if k != 255:
#         sum = 0
#         count = 0
#         for x in range(0, cols - 1):
#             for y in range(0, rows - 1):
#                 if (source_label[x, y]==k):
#                     sum += abs(log(1 / target_depth[x][y]) - log(1. / depth1[x][y]))
#                     count += 1
#         dict[k] = sum/count
# print(dict)
# sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}
# print(sorted_dict)
# classes_list = list(sorted_dict.keys())
# selected_classes = classes_list[:len(classes_list)//2 + 1]
# print(selected_classes)
# # classes = torch.from_numpy(np.array(selected_classes))
#
#
# MixMask = transformmasks.generate_class_mask(torch.from_numpy(source_label).cuda().long(), classes).unsqueeze(
#     0).cuda()
# np_arr = MixMask.cpu().detach().numpy()


class_mix(image1, source_label, depth1, target_image, target_depth)
# depth_mix_half_classes(image1, source_label, depth1, target_image, target_depth)
# depth_mix_half_classes_min_depth(image1, source_label, depth1, target_image, target_depth,MixMask)
# depth_mix_half_classes_median_depth(image1, source_label, depth1, target_image, target_depth,MixMask)



# def generate_depth_mask(depth, threshold):
#     if threshold.shape[0] == 1:
#         return depth.ge(threshold).float()
#     elif threshold.shape[0] == 2:
#         t1 = torch.min(threshold)
#         t2 = torch.max(threshold)
#         return depth.ge(t1).le(t2).float()
#     else:
#         raise NotImplementedError

#     for image_i in range(2):
#         generated_depth = depths[image_i]
#         min_depth = 0.1
#         max_depth = 0.4
#         depth_threshold = torch.rand(1, device=depths.device) * (max_depth - min_depth) + min_depth
#         if image_i == 0:
#             MixMask = generate_depth_mask(generated_depth, depth_threshold).cuda()
#         else:
#             MixMask = torch.cat(
#                 (MixMask, generate_depth_mask(generated_depth, depth_threshold).cuda()))
