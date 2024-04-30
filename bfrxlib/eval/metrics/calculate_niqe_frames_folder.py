from calculate_niqe import calculate_niqe_folder

class Args:
    def __init__(self, restored_folder, crop_border) -> None:
        self.restored_folder = restored_folder
        self.crop_border = crop_border

if __name__ == '__main__':
    import argparse
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--frames_folder', type=str, required=True, help='Path to the folder.')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    args = parser.parse_args()

    crop_border = args.crop_border
    all_dir = os.listdir(args.frames_folder)
    print('Total folder num: ', len(all_dir))

    niqe_all = []
    target_child = 'final_results'
    # target_child = 'restored_faces'
    target_child = ''
    for d in tqdm(all_dir):
        restored_folder = os.path.join(args.frames_folder, d, target_child)
        print(restored_folder)
        args_tmp = Args(restored_folder, crop_border)
        niqe = calculate_niqe_folder(args_tmp, verbose=False)
        niqe_all.append(niqe)

    result = sum(niqe_all) / len (niqe_all)
    print(args.frames_folder)
    print(f'Average of frames folder: NIQE: {result:.6f}')
    