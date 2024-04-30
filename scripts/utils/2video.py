import os
from tqdm import tqdm

from bfrxlib.preprocess.utils.video import images2video

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for frame_dir in tqdm(os.listdir(args.input_dir)):
        frame_dir_path = os.path.join(args.input_dir, frame_dir)
        video_name = frame_dir + '.mp4'
        video_path = os.path.join(args.output_dir, video_name)
        # print('frame_dir_path:', frame_dir_path, video_path)
        images2video(frame_dir_path, video_path, fps=args.fps, ffmpeg=args.ffmpeg)
        # exit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-suffix', '--suffix', type=str, default=None)
    parser.add_argument('-fps', '--fps', type=int, default=25)
    parser.add_argument('-ffmpeg', '--ffmpeg', action='store_true')
    args = parser.parse_args()
    main(args)
