import argparse
import itertools
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from mymodels import Generator_seg_ATSTNet
from mymodels import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from utils import MS_SSIM_Loss
from datasets import ImageDataset

from skimage.filters import threshold_otsu
from cppnet_models.cpp_net import CPPNet
from cppnet_predict import predict_each_image
from pretrained_pathoduet.vits_ATSTNet import VisionTransformerMoCo_ATSTNet


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/home/ubuntu/ATSTNet/dataset/dataset_root_512',
                    help='root directory of the H&E and IHC data')
parser.add_argument('--dataroot_HE_mask', type=str, default='/home/ubuntu/ATSTNet/dataset/dataset_root_512/HE_mask',
                    help='root directory of the mask of H&E (realA)')
parser.add_argument('--dataroot_IHC', type=str, default='/home/ubuntu/ATSTNet/dataset/dataset_root_512/IHC',
                    help='root directory of the corresponding IHC (realB) of H&E (realA)')
parser.add_argument('--dataroot_HE_pn_num_gt', type=str, default='/home/ubuntu/ATSTNet/dataset/dataset_root_512/HE_positive_nucleus_num_gt/positive_nucleus_num_gt_512.npz',
                    help='root directory of the positive nucleus num gt of H&E (realA)')
parser.add_argument('--modelroot', type=str, default='/home/ubuntu/ATSTNet/model_root/ATSTNet_main_train',
                    help='root directory of the model during training')
parser.add_argument('--checkpoint_path_HE', type=str, default='/home/ubuntu/ATSTNet/pretrained_pathoduet/pretrained_model/checkpoint_HE.pth',
                    help='root directory of the pretrained model')

parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--batchSize2', type=int, default=8, help='size of the batches for expert knowledge learning')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--continue_train', type=bool, default=False, help='load model and continue training')

opt = parser.parse_args()
print(opt)


def tensor2im(input_image, imtype=np.uint8):
    """ Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):                                 # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()                          # convert it into a numpy array
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0    # post-processing: transpose and scaling
    else:                                                                         # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def compute_l1_loss(a, b):
    """
    Compute the L1 loss between two torch.Tensors with gradient support.

    Args:
        a (torch.Tensor): First input tensor
        b (torch.Tensor): Second input tensor

    Returns:
        torch.Tensor: L1 loss value (scalar tensor with gradient)
    """
    # Ensure both tensors are of the same shape
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")

    # Compute L1 loss (mean absolute error)
    l1_loss = F.l1_loss(a, b, reduction='sum')

    return l1_loss


def compute_mse_loss(a, b):
    """
    Compute the Mean Square Error (MSE) loss between two torch.Tensor images.

    Args:
    a (torch.Tensor): First image tensor.
    b (torch.Tensor): Second image tensor.

    Returns:
    float: MSE loss.
    """
    # Ensure both tensors are of the same shape
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")

    # Compute MSE loss
    mse = torch.mean((a - b) ** 2)

    return mse


def rgb2hsi_tensor(image):
    """
    Function used to convert RGB to HSI.
    """
    r, g, b = image[0, :, :], image[1, :, :], image[2, :, :]     # Read channels.
    eps = 1e-6                                        # Avoid dividing zero.

    min_rgb = torch.minimum(torch.minimum(r, g), b)

    sum_rgb = r + g + b + eps
    img_s = 1.0 - 3.0 * min_rgb / sum_rgb       # Component S.

    min_val = torch.min(img_s)
    max_val = torch.max(img_s)
    img_s = (img_s - min_val) / (max_val - min_val + eps)

    return img_s


def rgb2hed_tensor(rgb):
    """
    RGB -> HED Conversion
    """
    # RGB to HED conversion matrix
    rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                 [0.07, 0.99, 0.11],
                                 [0.27, 0.57, 0.78]], device=rgb.device)

    hed_from_rgb = torch.inverse(rgb_from_hed)

    rgb = torch.clamp(rgb, min=1E-6)
    log_rgb = torch.log(rgb)
    log_adjust = torch.log(torch.tensor(1E-6, device=rgb.device))

    hed = torch.matmul(log_rgb.permute(1, 2, 0) / log_adjust, hed_from_rgb)
    hed = hed.permute(2, 0, 1)

    hed = torch.clamp(hed, min=0.0)

    return hed


def hed2rgb_tensor(hed):
    """
    HED -> RGB Conversion
    """
    rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                 [0.07, 0.99, 0.11],
                                 [0.27, 0.57, 0.78]], device=hed.device)

    log_adjust = -torch.log(torch.tensor(1E-6, device=hed.device))
    log_rgb = -torch.matmul(hed.permute(1, 2, 0) * log_adjust, rgb_from_hed)
    log_rgb = log_rgb.permute(2, 0, 1)
    rgb = torch.exp(log_rgb)

    rgb = torch.clamp(rgb, min=0.0, max=1.0)

    return rgb


def gaussian_blur_conv(input, kernel_size=3, sigma=1):

    channels = 1

    def get_gaussian_kernel(kernel_size, sigma):
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g

    g1d = get_gaussian_kernel(kernel_size, sigma).to(input.device)
    g2d = torch.outer(g1d, g1d)
    kernel = g2d.expand(channels, 1, kernel_size, kernel_size)

    input = input.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    blurred = F.conv2d(input, kernel, padding=kernel_size//2, groups=1)
    return blurred.squeeze()


class AuxiliaryLoss(nn.Module):
    def __init__(self, HE_positive_nucleus_num_gt, dataroot_HE_mask, dataroot_IHC):
        super(AuxiliaryLoss, self).__init__()
        self.HE_positive_nucleus_num_gt = HE_positive_nucleus_num_gt
        self.dataroot_HE_mask = dataroot_HE_mask
        self.dataroot_IHC = dataroot_IHC

    def forward(self, fake_B, batch, batchsize):
        losses_a1 = []
        losses_a2 = []
        losses_a3 = []
        losses_a4 = []
        losses_a5 = []

        for i in range(batchsize):

            # loss_a1: auxiliary task - global positive expression location matching
            fake_B_ten_o = fake_B[i]

            fake_B_ten_o.retain_grad()

            fake_B_ten = (fake_B_ten_o + 1) / 2

            fake_B_ten.retain_grad()

            ihc_hed = rgb2hed_tensor(fake_B_ten)

            null = torch.zeros_like(ihc_hed[0, :, :])
            ihc_h = hed2rgb_tensor(torch.stack((ihc_hed[0, :, :], null, null), dim=0))
            ihc_d = hed2rgb_tensor(torch.stack((null, null, ihc_hed[2, :, :]), dim=0))

            ihc_h.retain_grad()
            ihc_d.retain_grad()

            ihc_d_gray = 0.213 * ihc_d[0, :, :] + 0.715 * ihc_d[1, :, :] + 0.072 * ihc_d[2, :, :]

            filtered_img = gaussian_blur_conv(ihc_d_gray, kernel_size=3, sigma=1)
            filtered_img_np = filtered_img.detach().cpu().numpy()
            T = torch.tensor(threshold_otsu(filtered_img_np), device=filtered_img.device)

            k1 = 10.0
            mask_sigmoid = 1 - torch.sigmoid(k1 * (filtered_img - T))
            mask_hard = (mask_sigmoid > 0.5).float()
            mask = mask_hard - mask_sigmoid.detach() + mask_sigmoid

            kernel = torch.ones(3, 3, device=mask.device).unsqueeze(0).unsqueeze(0)
            k2 = 100.0
            mask_eroded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
            mask_eroded = 1 - torch.sigmoid(k2 * (8.5 - mask_eroded))
            mask_dilated = F.conv2d(mask_eroded, kernel, padding=1)
            mask_dilated = 1 - torch.sigmoid(k2 * (0.5 - mask_dilated))

            full_mask30 = F.max_pool2d(mask_dilated, kernel_size=29, stride=1, padding=14).squeeze()

            full_mask30.retain_grad()

            full_mask30_a1 = full_mask30

            filename_HE = batch['filename_HE'][i]
            HE_mask_gt = cv2.imread(os.path.join(self.dataroot_HE_mask, "HE_512", filename_HE), 0)
            HE_mask_gt = torch.tensor(HE_mask_gt, device=fake_B.device) / 255.0
            HE_mask_gt_copy_a4 = HE_mask_gt.clone()

            loss_a1 = compute_mse_loss(full_mask30_a1, HE_mask_gt)

            losses_a1.append(loss_a1)


            # loss_a2, loss_a3: auxiliary task - local positive expression location matching
            full_mask30_copy_a23 = full_mask30.clone()

            HE_mask_gt_copy_a23 = HE_mask_gt.clone()
            patch_size = 128
            num_patches = 4

            def get_patches(image):
                patches = []
                for i in range(num_patches):
                    for j in range(num_patches):
                        patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                        patches.append(patch)
                return patches

            full_mask30_copy_a23_patches = get_patches(full_mask30_copy_a23)
            HE_mask_gt_copy_a23_patches = get_patches(HE_mask_gt_copy_a23)

            threshold = 0.5
            counts = [torch.sum(patch > threshold) for patch in HE_mask_gt_copy_a23_patches]

            counts_tensor = torch.tensor(counts, device=full_mask30.device)

            max_indices = torch.argsort(counts_tensor)[-4:]
            min_indices = torch.argsort(counts_tensor)[:4]

            loss_a2 = 0
            loss_a3 = 0
            for idx in max_indices:
                loss_a2 += compute_mse_loss((full_mask30_copy_a23_patches[idx]), (HE_mask_gt_copy_a23_patches[idx]))
            for idx in min_indices:
                loss_a3 += compute_mse_loss((full_mask30_copy_a23_patches[idx]), (HE_mask_gt_copy_a23_patches[idx]))
            loss_a2 = loss_a2 / 4
            loss_a3 = loss_a3 / 4

            losses_a2.append(loss_a2)
            losses_a3.append(loss_a3)


            # loss_a4: auxiliary task - positive expression intensity matching
            full_mask30_copy_a4 = full_mask30.clone()

            fake_B_d_a4_o = ihc_d.clone()

            fake_B_d_a4 = fake_B_d_a4_o * 255.0

            background_mask_a4 = (full_mask30_copy_a4 == 0).unsqueeze(0).expand(3, -1, -1)

            fake_B_d_a4 = torch.where(background_mask_a4, torch.tensor(255.0, device=fake_B_d_a4.device), fake_B_d_a4)
            fake_B_d_a4 = fake_B_d_a4 / 255.0

            fake_B_d_a4_gray = 1 - (0.213 * fake_B_d_a4[0, :, :] + 0.715 * fake_B_d_a4[1, :, :] + 0.072 * fake_B_d_a4[2, :, :])

            foreground_mask = (full_mask30_copy_a4 == 1)
            full_mask30_copy_a4_255num = torch.sum(foreground_mask)
            fake_B_d_a4_gray_add = torch.sum(fake_B_d_a4_gray[foreground_mask])
            final_value_a4_fake = fake_B_d_a4_gray_add / full_mask30_copy_a4_255num if full_mask30_copy_a4_255num > 0 \
                                    else torch.tensor(0.0, device=fake_B_d_a4.device)

            IHC_gt = cv2.imread(os.path.join(self.dataroot_IHC, "IHC_512", filename_HE))
            IHC_gt = cv2.cvtColor(IHC_gt, cv2.COLOR_BGR2RGB)
            IHC_gt = torch.tensor(IHC_gt, device=fake_B.device) / 255.0
            IHC_gt = IHC_gt.permute(2, 0, 1)
            ihc_hed_a4 = rgb2hed_tensor(IHC_gt)
            ihc_d_a4 = hed2rgb_tensor(torch.stack((null, null, ihc_hed_a4[2, :, :]), dim=0))
            ihc_d_a4 = ihc_d_a4 * 255.0

            background_mask_gt = (HE_mask_gt_copy_a4 == 0).unsqueeze(0).expand(3, -1, -1)
            ihc_d_a4 = torch.where(background_mask_gt, torch.tensor(255.0, device=ihc_d_a4.device), ihc_d_a4)
            ihc_d_a4 = ihc_d_a4 / 255.0

            ihc_d_a4_gray = 1 - (0.213 * ihc_d_a4[0, :, :] + 0.715 * ihc_d_a4[1, :, :] + 0.072 * ihc_d_a4[2, :, :])

            foreground_mask_gt = (HE_mask_gt_copy_a4 == 1)
            HE_mask_gt_copy_a4_255num = torch.sum(foreground_mask_gt)
            ihc_d_a4_gray_add = torch.sum(ihc_d_a4_gray[foreground_mask_gt])
            final_value_a4_gt = ihc_d_a4_gray_add / HE_mask_gt_copy_a4_255num if HE_mask_gt_copy_a4_255num > 0 \
                                  else torch.tensor(0.0, device=fake_B_d_a4.device)

            loss_a4 = compute_mse_loss(final_value_a4_fake, final_value_a4_gt)

            losses_a4.append(loss_a4)


            # loss_a5: auxiliary task - nucleus number matching in positive regions
            full_mask30_copy_a5 = full_mask30.clone()
            fake_B_h_a5 = ihc_h.clone()

            fake_B_h_a5.retain_grad()

            fake_B_h_a5 = fake_B_h_a5 * 255.0
            background_mask_a5 = (full_mask30_copy_a5 == 0).unsqueeze(0).expand(3, -1, -1)
            fake_B_h_a5 = torch.where(background_mask_a5, torch.tensor(255.0, device=ihc_d_a4.device), fake_B_h_a5)
            fake_B_h_a5 = fake_B_h_a5 / 255.0
            s = rgb2hsi_tensor(fake_B_h_a5)

            num_100h = torch.sum(s > (100 / 255.0))

            with torch.no_grad():

                s_np = (s.cpu().numpy() * 255).astype(np.uint8)
                HE_positive_nucleus_num = 0
                if num_100h.item() > 30:
                    print("filename_HE: ", filename_HE, "CPP-Net Predicting-----------")
                    n_sampling = 6
                    nc_in = 1
                    n_rays = 32
                    model_weight_path = r'/home/ubuntu/ATSTNet/cppnet_checkpoint/IHC_nucleus_stage2/CHECKPOINT.t7'
                    center_prob_thres = 0.4
                    seg_prob_thres = 0.5
                    erosion_factor_list = [float(i + 1) / n_sampling for i in range(n_sampling)]
                    model_dist = CPPNet(nc_in, n_rays, erosion_factor_list=erosion_factor_list).cuda()
                    model_dist.load_state_dict(torch.load(model_weight_path))
                    model_dist.eval()
                    inst_map = predict_each_image(
                        model_dist, s_np, (0, 1),
                        center_prob_thres=center_prob_thres, seg_prob_thres=seg_prob_thres, n_rays=n_rays
                    )
                    HE_positive_nucleus_num = np.max(inst_map)
                    print("filename_HE: ", filename_HE, "CPP-Net Prediction Over-----------")

            HE_positive_nucleus_num = torch.tensor(HE_positive_nucleus_num, device=fake_B.device, dtype=torch.float32)
            positive_nucleus_num_gt = torch.tensor(self.HE_positive_nucleus_num_gt[filename_HE].item(), device=fake_B.device, dtype=torch.float32)

            loss_a5 = compute_l1_loss(HE_positive_nucleus_num, positive_nucleus_num_gt)
            loss_a5 = loss_a5 / 100.0

            losses_a5.append(loss_a5)


        loss_a1_realA2B = torch.stack(losses_a1).mean()
        loss_a2_realA2B = torch.stack(losses_a2).mean()
        loss_a3_realA2B = torch.stack(losses_a3).mean()
        loss_a4_realA2B = torch.stack(losses_a4).mean()
        loss_a5_realA2B = torch.stack(losses_a5).mean()

        return loss_a1_realA2B, loss_a2_realA2B, loss_a3_realA2B, loss_a4_realA2B, loss_a5_realA2B

if __name__ == '__main__':

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = VisionTransformerMoCo_ATSTNet(pretext_token=True, global_pool='avg')
    netG_B2A = Generator_seg_ATSTNet(opt.output_nc, opt.input_nc, 10, alt_leak=False, neg_slope=0.1)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    if opt.cuda:
        netG_A2B = netG_A2B.cuda()
        netG_B2A = netG_B2A.cuda()
        netD_A = netD_A.cuda()
        netD_B = netD_B.cuda()

    netG_A2B.train()
    netG_B2A.train()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_HE = torch.load(opt.checkpoint_path_HE, map_location=device)
    netG_A2B.load_state_dict(checkpoint_HE, strict=False)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, batch_size=opt.batchSize, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    ###################################
    start_epoch = opt.epoch

    HE_positive_nucleus_num_gt = np.load(opt.dataroot_HE_pn_num_gt)
    auxiliary_loss = AuxiliaryLoss(HE_positive_nucleus_num_gt, opt.dataroot_HE_mask, opt.dataroot_IHC)

    loss_G_list = []
    loss_identity_list = []
    loss_GAN_list = []
    loss_cycle_list = []
    loss_cycle_ssim_list = []
    loss_a1_list = []
    loss_a2_list = []
    loss_a3_list = []
    loss_a4_list = []
    loss_a5_list = []
    loss_D_A_list = []
    loss_D_B_list = []
    ###### Training ######
    for epoch in range(start_epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            print("epoch-i: ", epoch, "-", i)

            real_A = Variable(input_A.copy_(batch['HE']))
            real_B = Variable(input_B.copy_(batch['IHC']))

            # Generators A2B and B2A
            optimizer_G.zero_grad()

            def netG_A2B_process(real_A, netG_A2B):
                crop_coords = [
                    (0, 0), (0, 144), (0, 288),
                    (144, 0), (144, 144), (144, 288),
                    (288, 0), (288, 144), (288, 288)]
                fake_B = torch.zeros_like(real_A)
                crop_size = 224
                for y, x in crop_coords:
                    crop = real_A[:, :, y:y + crop_size, x:x + crop_size]
                    fake_crop = netG_A2B(crop)

                    fake_B[:, :, y:y + crop_size, x:x + crop_size] = fake_crop

                return fake_B

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B_process(real_B, netG_A2B)
            loss_identity_B = criterion_identity(same_B, real_B)
            # G_B2A(A) should equal A if real A is fed
            same_A, _, _, _ = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)

            # GAN loss
            fake_B = netG_A2B_process(real_A, netG_A2B)

            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A, latent_feature_fa, out_pathology_path_fa, pathology_feature_fa = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A, latent_feature_ra, out_pathology_path_ra, pathology_feature_ra = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
            loss_cycle_ssim_ABA = criterion_ssim(recovered_A, real_A)

            recovered_B = netG_A2B_process(fake_A, netG_A2B)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)
            loss_cycle_ssim_BAB = criterion_ssim(recovered_B, real_B)

            # loss_a1, loss_a2, loss_a3, loss_a4, loss_a5: auxiliary tasks-assisted loss
            loss_a1_realA2B, loss_a2_realA2B, loss_a3_realA2B, loss_a4_realA2B, loss_a5_realA2B = auxiliary_loss(fake_B, batch, opt.batchSize)

            # Total loss
            loss_G = 10.0 * (loss_identity_B + loss_identity_A) + \
                     1.0 * (loss_GAN_A2B + loss_GAN_B2A) + \
                     5.0 * (loss_cycle_ABA + loss_cycle_BAB) + \
                     5.0 * (loss_cycle_ssim_ABA + loss_cycle_ssim_BAB) + \
                     50.0 * loss_a1_realA2B + \
                     100.0 * loss_a2_realA2B + \
                     100.0 * loss_a3_realA2B + \
                     1.0 * loss_a4_realA2B + \
                     100.0 * loss_a5_realA2B

            loss_G.backward()

            optimizer_G.step()

            ###################################

            # Discriminator A
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_Ad = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_Ad.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            loss_D_A.backward()

            optimizer_D_A.step()

            ###################################

            # Discriminator B
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_Bd = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_Bd.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            loss_D_B.backward()

            optimizer_D_B.step()

            ###################################

            loss_D_A_list.append(loss_D_A.item())
            loss_D_B_list.append(loss_D_B.item())

            # save models at half of an epoch
            if i == 200 and epoch == 1:
                saveroot = os.path.join(opt.modelroot, 'star_temp')
                if not os.path.exists(saveroot):
                    os.makedirs(saveroot)

                # Save models checkpoints
                netG_A2B_checkpoints = {
                    "model": netG_A2B.state_dict()
                }
                torch.save(netG_A2B_checkpoints, os.path.join(saveroot, 'netG_A2B.pth'))

                netG_B2A_checkpoints = {
                    "model": netG_B2A.state_dict(),
                    'optimizer': optimizer_G.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': lr_scheduler_G.state_dict()
                }
                torch.save(netG_B2A_checkpoints, os.path.join(saveroot, 'netG_B2A.pth'))

                netD_A_checkpoints = {
                    "model": netD_A.state_dict(),
                    'optimizer': optimizer_D_A.state_dict(),
                    'lr_schedule': lr_scheduler_D_A.state_dict()
                }
                torch.save(netD_A_checkpoints, os.path.join(saveroot, 'netD_A.pth'))

                netD_B_checkpoints = {
                    "model": netD_B.state_dict(),
                    'optimizer': optimizer_D_B.state_dict(),
                    'lr_schedule': lr_scheduler_D_B.state_dict()
                }
                torch.save(netD_B_checkpoints, os.path.join(saveroot, 'netD_B.pth'))

        print('loss_G: ', loss_G)
        print('loss_D_A: ', loss_D_A)
        print('loss_D_B: ', loss_D_B)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        saveroot = os.path.join(opt.modelroot, 'epoch' + str(epoch))
        if not os.path.exists(saveroot):
            os.makedirs(saveroot)

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), os.path.join(saveroot, 'netG_A2B.pth'))
        torch.save(netG_B2A.state_dict(), os.path.join(saveroot, 'netG_B2A.pth'))
        torch.save(netD_A.state_dict(), os.path.join(saveroot, 'netD_A.pth'))
        torch.save(netD_B.state_dict(), os.path.join(saveroot, 'netD_B.pth'))
