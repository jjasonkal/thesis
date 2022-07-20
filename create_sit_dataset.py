import numpy as np
import os
# from torchvision.utils import save_image
import torch.nn as nn
import torch
from PIL import Image
import argparse
from matplotlib import pyplot as plt
import sys
import datetime

from torchvision.utils import save_image
from tqdm import tqdm
import random
from core.config import get_arguments, post_config
# parser = get_arguments()
# opt = parser.parse_args()
# opt = post_config(opt)
from core.functions import norm_image
from data_handlers import CreateSrcDataLoader
from core.training import concat_pyramid_eval
# import imageio
# import Logger
from torchvision import transforms

from configs.global_vars import IMG_MEAN
from utils.visualization import DeNormalize

# scale_factor = 0.5
# trained_procst_path = "/mnt/genesis/kaltsikis/data/pretrained_procst_gta2cs.pth"
# model = torch.load(trained_procst_path)
# model.eval()
# model = torch.nn.DataParallel(model)
# print('Model loaded')
# images_path = "/mnt/genesis/kaltsikis/data/gta/images/"
# images_names = [file for file in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, file))]
# print('Number of images to convert: %d' % len(images_names))
# image1_path = images_names[0]
# image = Image.open(images_path+image1_path).convert('RGB')
# plt.imshow(image)
# plt.show()
torch.cuda.empty_cache()

parser = get_arguments()
opt = parser.parse_args()
opt = post_config(opt)
model = torch.load(opt.trained_procst_path)
for i in range(len(model)):
    model[i].eval()
    model[i] = torch.nn.DataParallel(model[i])
    model[i].to(opt.device)
opt.num_scales = opt.curr_scale = len(model)-1
source_train_loader = CreateSrcDataLoader(opt, get_filename=True, get_original_image=True)
if opt.skip_created_files:
    already_created = next(os.walk(opt.sit_output_path))[2]
    for f in already_created:
        if f in source_train_loader.dataset.img_ids:
            source_train_loader.dataset.img_ids.remove(f)
print('Number of images to convert: %d' % len(source_train_loader.dataset.img_ids))
with torch.no_grad():
    for source_scales, filenames in tqdm(source_train_loader):
        # for i in range(len(source_scales)):
        #     source_scales[i] = source_scales[i].to(opt.device)
        sit_batch = concat_pyramid_eval(model, source_scales, opt)
        for i, filename in enumerate(filenames):
            save_image(norm_image(sit_batch[i]), os.path.join(opt.sit_output_path, filename))
print('Finished Creating SIT Dataset.')

# model.to(opt.device)
# source_train_loader = CreateSrcDataLoader(opt, get_filename=True, get_original_image=True)

# if opt.skip_created_files:
#     already_created = next(os.walk(opt.sit_output_path))[2]
#     for f in already_created:
#         if f in source_train_loader.dataset.img_ids:
#             source_train_loader.dataset.img_ids.remove(f)

# print('Number of images to convert: %d' % len(source_train_loader.dataset.img_ids))
# for source_scales, filenames in tqdm(source_train_loader):
#     for i in range(len(source_scales)):
#         source_scales[i] = source_scales[i].to(opt.device)
#     sit_batch = concat_pyramid_eval(model, source_scales, opt)
#     for i, filename in enumerate(filenames):
#         save_image(norm_image(sit_batch[i]), os.path.join(opt.sit_output_path, filename))
# print('Finished Creating SIT Dataset.')

# if __name__ == "__main__":

