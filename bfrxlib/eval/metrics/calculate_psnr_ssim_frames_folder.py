from calculate_psnr_ssim import calculate_psnr_ssim


class Args:
    def __init__(self, restored_folder, gt_folder, crop_border=0, test_y_channel=False, correct_mean_var=True) -> None:
        self.restored_folder = restored_folder
        self.gt_folder = gt_folder
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self.correct_mean_var = correct_mean_var


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    args = parser.parse_args()

    all_restored_frames = os.listdir(args.restored_folder)

    psnrs = []
    ssims = []
    for restored_frames_path in all_restored_frames:
        input_path = os.path.join(args.restored_folder, restored_frames_path)
        # input_path = os.path.join(args.restored_folder, restored_frames_path, 'final_results')
        gt_path = os.path.join(args.gt_folder, restored_frames_path)
        psnr_res, ssim_res = calculate_psnr_ssim(Args(input_path, gt_path), verbose=False)
        psnrs.append(psnr_res)
        ssims.append(ssim_res)

    print('Average PSNR: ', sum(psnrs) / len(psnrs))
    print('Average SSIM: ', sum(ssims) / len(ssims))