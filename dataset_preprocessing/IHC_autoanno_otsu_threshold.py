# Automatic H&E pseudo-labeling from IHC serial sections

import os
import glob
from skimage import io
from skimage.color import rgb2hed, hed2rgb, rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import closing, disk
import numpy as np
import cv2
from skimage.util import img_as_ubyte

# Input & output folder
input_folder = r'/home/ubuntu/ATSTNet/dataset_preprocessing/input_folder'
output_folder = r'/home/ubuntu/ATSTNet/dataset_preprocessing/output_folder'

# Create output folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = glob.glob(os.path.join(input_folder, '*.png'))

for file_path in image_files:

    print("file_path: ", file_path)

    # read image
    input_img = io.imread(file_path)

    # HED color deconvolution
    ihc_hed = rgb2hed(input_img)

    # separate each stain into RGB
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    base_name = os.path.basename(file_path)

    # convert stain RGB to grayscale
    ihc_h_gray = rgb2gray(ihc_h)
    ihc_d_gray = rgb2gray(ihc_d)

    # output_path_hema = os.path.join(output_folder, base_name.replace('.', '_hema.'))
    # io.imsave(output_path_hema, img_as_ubyte(ihc_h))
    # output_path_dab = os.path.join(output_folder, base_name.replace('.', '_dab.'))
    # io.imsave(output_path_dab, img_as_ubyte(ihc_d))
    # output_path_hema_gray = os.path.join(output_folder, base_name.replace('.', '_hema_gray.'))
    # io.imsave(output_path_hema_gray, img_as_ubyte(ihc_h_gray))
    # output_path_dab_gray = os.path.join(output_folder, base_name.replace('.', '_dab_gray.'))
    # io.imsave(output_path_dab_gray, img_as_ubyte(ihc_d_gray))

    # Gaussian smoothing
    filtered_img = gaussian(ihc_d_gray, sigma=1)

    # Otsu threshold
    T = threshold_otsu(filtered_img)
    mask = 1 - (filtered_img > T)

    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel
    mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Morphological closing
    full_mask30 = closing(mask, disk(30))
    full_mask30 = (full_mask30 * 255).astype(np.uint8)
    full_mask30[full_mask30 > 0] = 255

    base_name = os.path.basename(file_path)
    output_path_1 = os.path.join(output_folder, base_name)
    output_path_2 = os.path.join(output_folder, base_name.replace('.png', '_filtered.png'))
    output_path_3 = os.path.join(output_folder, base_name.replace('.png', '_mask.png'))
    output_path_4 = os.path.join(output_folder, base_name.replace('.png', '_full_disk30.png'))

    io.imsave(output_path_1, input_img)
    io.imsave(output_path_2, (filtered_img * 255).astype(np.uint8))
    io.imsave(output_path_3, (mask * 255).astype(np.uint8))
    io.imsave(output_path_4, full_mask30)
