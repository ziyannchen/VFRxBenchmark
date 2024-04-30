from calculate_lpips import calculate_lpips


class Args:
    def __init__(self, restored_folder, gt_folder) -> None:
        self.restored_folder = restored_folder
        self.gt_folder = gt_folder

if __name__ == '__main__':
    import os
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    args = parser.parse_args()

    all_restored_frames = sorted(os.listdir(args.restored_folder))

    lpipses = []
    for restored_frames_path in all_restored_frames:
        # input_path = os.path.join(args.restored_folder, restored_frames_path, 'final_results')
        input_path = os.path.join(args.restored_folder, restored_frames_path)
        gt_path = os.path.join(args.gt_folder, restored_frames_path)
        # print(input_path, gt_path)
        lpips = calculate_lpips(Args(input_path, gt_path), verbose=False)
        lpipses.append(lpips)

    print('Average LPIPS: ', sum(lpipses) / len(lpipses))