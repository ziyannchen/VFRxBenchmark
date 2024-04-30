import glob
import os


def crop_from_meta(meta_path, data_path):
    with open(meta_f, 'r') as f:
        for line in f:
            # get the coordinates of face
            if line.startswith('CROP'):
                clip_crop_bbox = line.strip().split(' ')[-4:]
                x0 = int(clip_crop_bbox[0])
                y0 = int(clip_crop_bbox[1])
                x1 = int(clip_crop_bbox[2])
                y1 = int(clip_crop_bbox[3])
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--meta_folder', type=str)
    parser.add_argument('-d', '-data_folder', type=str)
    args = parser.parse_args()

    for meta_f in sorted(glob.glob(os.path.join(args.meta_path, '*'))):
        pass