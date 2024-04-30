class Args:
    def __init__(self, restored_folder, gt_folder, test_model_path='experiments/pretrained_models/metric_weights/resnet18_110.pth') -> None:
        self.restored_folder = restored_folder
        self.gt_folder = gt_folder
        self.test_model_path = test_model_path


if __name__ == '__main__':
    import argparse
    import os

    from calculate_cos_dist import calculate_cos_dist

    parser = argparse.ArgumentParser()
    parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-gt_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument('-test_model_path', type=str, default='experiments/pretrained_models/metric_weights/resnet18_110.pth')
    args = parser.parse_args()

    all_frames_folder = os.listdir(args.restored_folder)

    idps = []
    for frames_folder in all_frames_folder:
        input_path = os.path.join(args.restored_folder, frames_folder)
        gt_path = os.path.join(args.gt_folder, frames_folder)
        idp = calculate_cos_dist(Args(input_path, gt_path, args.test_model_path), verbose=False)
        idps.append(idp)

    result = sum(idps) / len(idps)
    print('Average ID Preservation: ', result)