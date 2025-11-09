import numpy as np
from csbdeep.utils import normalize
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label

import torch
import math
import warnings
warnings.filterwarnings("ignore")

# try:
#     import numpy_gpu as npgpu
# except:
#     print('The package "numpy_gpu" is used for comparison only. You can try to use the package "numpy_gpu", but it is not necessary.')


ap_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def predict_each_image(
        model_dist, img,
        axis_norm=[0, 1], center_prob_thres=0.4, seg_prob_thres=0.5,
        n_rays=32, FPP=True, sin_angles=None, cos_angles=None, dist_cmp='cuda'
):
    division = 1

    img = img.copy()
    img = normalize(img, 1, 99.8, axis=axis_norm)

    h, w = img.shape
    if h % division != 0 or w % division != 0:
        dh = (h // division + 1) * division - h
        dw = (w // division + 1) * division - w
        img = np.pad(img, ((0, dh), (0, dw)), 'constant')

    assert (dist_cmp in ['cuda', 'cpu', 'np', 'npcuda'])

    input = torch.tensor(img)
    input = input.unsqueeze(0).unsqueeze(0)
    preds = model_dist(input.cuda())
    dist_cuda = preds[0][-1][:, :, :h, :w]
    dist = dist_cuda.data.cpu()
    prob = preds[1][-1].data.cpu()[:, :, :h, :w]
    seg = preds[2][-1].data.cpu()[:, :, :h, :w]

    dist_numpy = dist.numpy().squeeze()  # dist_numpy: 32, 512, 512
    prob_numpy = prob.numpy().squeeze()  # prob_numpy: 512, 512
    seg = seg.numpy().squeeze()  # seg: 512, 512
    prob_numpy = prob_numpy * seg  # prob_numpy: 512, 512  # (seg>=seg_prob_thres).astype(np.float32)

    dist_numpy = np.transpose(dist_numpy, (1, 2, 0))  # dist_numpy: 512, 512, 32
    coord = dist_to_coord(dist_numpy)  # coord: 512, 512, 2, 32    # 来自原始stardist=0.6.0
    points = non_maximum_suppression(coord, prob_numpy,
                                     prob_thresh=center_prob_thres)  # 从新版本的stardist=0.8.5里的nms引入，因此使用老版本
    star_label = polygons_to_label(coord, prob_numpy, points)  # 来自原始stardist=0.6.0

    # st0 = time.time()
    # You can try different approaches to finish the process of distance calculation. In our experiments dist_cmp='cuda' seems faster
    if FPP and sin_angles is None:
        if dist_cmp == 'cuda':
            angles = torch.arange(n_rays).float() / float(n_rays) * math.pi * 2.0  # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1)
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1)
            sin_angles = sin_angles.cuda()
            cos_angles = cos_angles.cuda()

            offset_ih = sin_angles * dist_cuda
            offset_iw = cos_angles * dist_cuda
            # 1, r, h, w, 2
            offsets = torch.stack([offset_iw, offset_ih], dim=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(dim=1).data.cpu().squeeze(dim=0).numpy()).astype(np.int16)
        elif dist_cmp == 'cpu':
            angles = torch.arange(n_rays).float() / float(n_rays) * math.pi * 2.0  # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1)
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1)

            offset_ih = sin_angles * dist
            offset_iw = cos_angles * dist
            # 1, r, h, w, 2
            offsets = torch.stack([offset_iw, offset_ih], dim=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(dim=1).data.cpu().squeeze(dim=0).numpy()).astype(np.int16)
        elif dist_cmp == 'np':
            angles = torch.arange(n_rays).float() / float(n_rays) * math.pi * 2.0  # 0 - 2*pi
            sin_angles = torch.sin(angles).view(1, n_rays, 1, 1).data.numpy()
            cos_angles = torch.cos(angles).view(1, n_rays, 1, 1).data.numpy()

            offset_ih = sin_angles * dist.numpy()
            offset_iw = cos_angles * dist.numpy()
            # 1, r, h, w, 2
            offsets = np.stack([offset_iw, offset_ih], axis=-1)
            # h, w, 2
            mean_coord = np.round(offsets.mean(axis=1).squeeze(axis=0)).astype(np.int16)
        # elif dist_cmp == 'npcuda':
        #     angles = torch.arange(n_rays).float() / float(n_rays) * math.pi * 2.0  # 0 - 2*pi
        #     sin_angles = torch.sin(angles).view(1, n_rays, 1, 1).data.numpy()
        #     cos_angles = torch.cos(angles).view(1, n_rays, 1, 1).data.numpy()
        #
        #     offset_ih = npgpu.dot(sin_angles, dist.numpy())
        #     offset_iw = npgpu.dot(cos_angles, dist.numpy())
        #     # 1, r, h, w, 2
        #     offsets = np.stack([offset_iw, offset_ih], axis=-1)
        #     # h, w, 2
        #     mean_coord = np.round(offsets.mean(axis=1).squeeze(axis=0)).astype(np.int16)

    pred = star_label

    # Offset-based Post Processing:
    if FPP:
        seg_remained = np.logical_and(seg >= seg_prob_thres, pred == 0)
        while seg_remained.any():
            if seg_remained.any():
                rxs, rys = np.where(seg_remained)
                mean_coord_remained = mean_coord[seg_remained, :]
                pred_0 = pred.copy()
                rxs_a = np.clip((rxs + mean_coord_remained[:, 1]).astype(np.int16), 0, h - 1)
                rys_a = np.clip((rys + mean_coord_remained[:, 0]).astype(np.int16), 0, w - 1)
                pred[seg_remained] = pred[(rxs_a, rys_a)]
                if not ((pred_0 != pred).any()):
                    break
            else:
                break
            seg_remained = np.logical_and(seg >= seg_prob_thres, pred == 0)

    return pred