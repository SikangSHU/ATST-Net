# Steps for generating the nucleus number (GT) in positive regions.

import os
import cv2
import numpy as np


def rgb2hsi(image):
    # Function used to convert RGB to HSI.
    r, g, b = cv2.split(image.astype(np.float32))     # Read channels.
    eps = 1e-6                                        # Avoid dividing zero.

    min_rgb = cv2.min(r, cv2.min(b, g))
    img_s = 1 - 3 * min_rgb / (r + g + b + eps)       # Component S.

    temp_s = img_s - np.min(img_s)
    img_s = temp_s / np.max(temp_s)

    return img_s


# Step 1: Filtering out the positive regions.
print("Part1 begin.---------------------------")
path_in_1 = r'/home/ubuntu/ATSTNet/dataset/dataset_root_512/HE_positive_nucleus_num_gt/hema_channel'
path_in_2 = r'/home/ubuntu/ATSTNet/dataset/dataset_root_512/HE_positive_nucleus_num_gt/mask_channel'
path_grey_out = r'/home/ubuntu/ATSTNet/dataset/dataset_root_512/HE_positive_nucleus_num_gt/grey_hema_channel_with_mask'
# TODO: To be modified
file_list_in = os.listdir(path_in_1)
for filename in file_list_in:
    I_1 = cv2.imread(path_in_1 + "/" + filename)
    I_2 = cv2.imread(path_in_2 + "/" + filename, 0)
    I_1_rgb = cv2.cvtColor(I_1, cv2.COLOR_BGR2RGB)
    I_1_rgb[I_2 == 0] = [255, 255, 255]
    s = rgb2hsi(I_1_rgb)
    s = (s * 255.0).astype(np.uint8)

    cv2.imwrite(path_grey_out + "/" + filename, s)
print("Part1 complete.---------------------------")

# Step 2: Predicting nucleus number (GT) in positive regions by CPP-Net.
