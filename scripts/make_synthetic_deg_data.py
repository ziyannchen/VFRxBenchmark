if __name__ == '__main__':
    import os
    import cv2
    import glob
    import numpy as np
    from tqdm import tqdm
    import argparse

    from bfrxlib.preprocess.utils import Degrader

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--out_dir', type=str)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    

    args_setting = {
        'blur_kernel_size': 41,
        'kernel_list': ['iso', 'aniso'],
        'kernel_prob': [0.5, 0.5],
        'blur_sigma': [0.1, 15],
        'downsample_range': [0.8, 32],
        'noise_range': [0, 20],
        'jpeg_range': [30, 100]
    }
    degrader = Degrader(**args_setting)

    HQ_imgs = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))
    for index in tqdm(range(len(HQ_imgs))):
        filename = os.path.basename(HQ_imgs[index])  
        # print(filename)
        img_gt = cv2.imread(HQ_imgs[index], cv2.IMREAD_COLOR)
        # cv2.imwrite(os.path.join(out_dir, filename), img_gt)
        img_gt = img_gt.astype(np.float32)/255.
        img_lq = degrader.degrade_process(img_gt)
        cv2.imwrite(os.path.join(args.out_dir, filename), img_lq)