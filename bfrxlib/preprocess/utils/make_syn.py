import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F

# import basicsr.data.degradations as degradations
import .degradations

class Degrader(object):
    def __init__(self, kernel_list = ['iso', 'aniso'], kernel_prob = [0.5, 0.5], blur_kernel_size = 41,
                blur_sigma = [0.1, 10], downsample_range = [0.8, 8], noise_range = [0, 20], jpeg_range = [60, 100]):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10] # 1, 
        self.downsample_range = [0.8, 8] # 1, 30
        self.noise_range = [0, 20] # 0, 20
        self.jpeg_range = [60, 100] # 30, 90
    
    def degrade_process(self, img_gt):
        # if random.random() > 0.5:
        #     img_gt = cv2.flip(img_gt, 1)

        h, w = img_gt.shape[:2]
       
        # random color jitter 
        # if np.random.uniform() < self.color_jitter_prob:
        #     jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
        #     img_gt = img_gt + jitter_val
        #     img_gt = np.clip(img_gt, 0, 1)    

        # # random grayscale
        # if np.random.uniform() < self.gray_prob:
        #     img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        #     img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        # img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_lq

if __name__ == '__main__':
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    HQ_imgs = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))

    degrader = Degrader()
    for index in tqdm(range(len(HQ_imgs))):
        filename = os.path.basename(HQ_imgs[index])  
        # print(filename)
        img_gt = cv2.imread(HQ_imgs[index], cv2.IMREAD_COLOR)
        # cv2.imwrite(os.path.join(out_dir, filename), img_gt)
        img_gt = img_gt.astype(np.float32)/255.
        img_lq = degrader.degrade_process(img_gt)
        cv2.imwrite(os.path.join(args.out_dir, filename), img_lq)