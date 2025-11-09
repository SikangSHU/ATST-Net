import argparse
import os
import time
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pretrained_pathoduet.vits_ATSTNet import VisionTransformerMoCo_ATSTNet


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestImage(Dataset):
    def __init__(self, image, positions, transform, patch_size=512):
        self.image = image
        self.positions = positions
        self.transform = transform
        self.patch_size = patch_size

    def __getitem__(self, index):
        x, y = self.positions[index]
        patch = self.image[x - self.patch_size:x, y - self.patch_size:y, :]
        patch = self.transform(patch)
        return {'A': patch, 'x': x, 'y': y}

    def __len__(self):
        return len(self.positions)

# 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--generator_A2B', type=str)
    parser.add_argument('--test_data_path', type=str)
    opt = parser.parse_args()

    opt.generator_A2B = r'/home/ubuntu/ATSTNet/model_root/ATSTNet_main_train/epoch39/netG_A2B.pth'
    opt.test_data_path = r'/home/ubuntu/ATSTNet/dataset/test_512'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])

    model = VisionTransformerMoCo_ATSTNet(pretext_token=True, global_pool='avg')
    model.load_state_dict(torch.load(opt.generator_A2B, map_location=device))
    model.eval().to(device)

    def infer_patches(img_tensor):
        crop_coords = [(i, j) for i in [0, 144, 288] for j in [0, 144, 288]]
        crop_size = 224
        fake_B = torch.zeros_like(img_tensor)
        crops = [img_tensor[:, :, y:y + crop_size, x:x + crop_size] for y, x in crop_coords]
        crops_tensor = torch.cat(crops, dim=0)
        fake_crops = model(crops_tensor)
        for idx, (y_, x_) in enumerate(crop_coords):
            fake_B[:, :, y_:y_ + crop_size, x_:x_ + crop_size] = fake_crops[idx:idx + 1]
        return fake_B

    for image_name in os.listdir(opt.test_data_path):
        filepath = os.path.join(opt.test_data_path, image_name)
        if not os.path.isfile(filepath):
            continue

        print(f"Processing: {image_name}")
        img = Image.open(filepath).convert('RGB')
        img_np = np.array(img)
        h, w, _ = img_np.shape

        assert h == opt.size and w == opt.size, f"Input image must be {opt.size}x{opt.size}, but got {h}x{w}"

        positions = [(opt.size, opt.size)]

        dataset = TestImage(img_np, positions, transform, patch_size=opt.size)
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                real_A = batch['A'].to(device)

                start_time = time.perf_counter()
                fake_B = infer_patches(real_A)
                torch.cuda.synchronize()
                print(f"Batch {i+1}/{len(dataloader)} inference time: {time.perf_counter() - start_time:.4f}s")

                fake_B = (0.5 * (fake_B + 1.0)).clamp(0, 1)

                for j in range(real_A.size(0)):
                    out_np = (fake_B[j] * 255).byte().permute(1, 2, 0).cpu().numpy()
                    save_dir = os.path.join(os.path.dirname(opt.generator_A2B), 'output/A2B', os.path.basename(opt.test_data_path))
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, os.path.splitext(image_name)[0] + '_pre_B.png')
                    Image.fromarray(out_np).save(save_path)
                    print(f"Saved to {save_path}\n")

if __name__ == '__main__':
    main()
