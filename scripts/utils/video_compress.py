import cv2
import ffmpeg
import glob
import numpy as np
import os
import random
from tqdm import tqdm

from basicsr.utils import imwrite
from basicsr.data.data_util import read_img_seq

source_dir = '/cpfs01/shared/xpixel/celebv-text/frames_all'
save_path = 'Celebv-Text/ffmpeg'
tmp_file = f'{save_path}/tmp2.mp4'
os.makedirs(save_path, exist_ok=True)
resized_width = resized_height = 512
format = 'h264'
loglevel = 'error'
fps = 40
all_dir = os.listdir(source_dir)
# print(all_dir[0], len(all_dir[::-1]))
for i, d in enumerate(tqdm([all_dir[0]]+all_dir[::-1])):
    save_img_path = os.path.join(save_path, d)
    if os.path.exists(save_img_path):
        continue
    os.makedirs(save_img_path, exist_ok=True)

    extra_args = dict()
    if format == 'h264':
        vcodec = 'libx264'
        profile = random.choices(['baseline', 'main', 'high'], [0.1, 0.2, 0.7])[0]
        extra_args['profile:v'] = profile
    crf = np.random.uniform(24, 32)

    cmd_video_compress = f'ffmpeg -framerate {fps} -pattern_type glob -i \'{source_dir}/{d}/*.jpg\' -y -crf {crf} -pix_fmt yuv420p -profile:v {profile} {tmp_file}'
    print(cmd_video_compress)
    os.system(cmd_video_compress)

    cap = cv2.VideoCapture(f'{tmp_file}')
    count = 0
    success = True
    while success:
        # success判断获取下一帧是否成功
        success, img = cap.read()
        # print(count, interval, count % interval)
        if img is not None:
            #帧尺寸resize到512x512
            resized_img = img
            # resized_img = cv2.resize(img, (resized_height, resized_width), interpolation=cv2.INTER_LINEAR)
            # save frame as JPEG file
            cv2.imwrite(save_img_path+'/'+d[:-4]+f'_{str(count).zfill(4)}.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            count += 1
        if count >= 301:
            break

    # img_lqs = []
    # img_list = glob.glob(os.path.join(source_dir, d, '*'))
    # for img_lq_path in img_list:
    #     img_lq = cv2.imread(img_lq_path, cv2.IMREAD_COLOR)
    #     img_lqs.append(img_lq)
    # print(len(img_lqs))

    # ffmpeg_img2video = (
    #     ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{resized_width}x{resized_height}', r=fps)
    #           .filter('fps', fps=fps, round='up')
    #           .output('pipe:', format=format, pix_fmt='yuv420p', crf=crf, vcodec=vcodec, **extra_args))
            #   .global_args('-hide_banner').global_args('-loglevel', loglevel)
            #   .run_async(pipe_stdin=True, pipe_stdout=True))
    # ffmpeg_video2img = (
    #     ffmpeg.input('pipe:', format=format)
    #     .output('pipe:', format='rawvideo',pix_fmt='rgb24')
    #     .global_args('-hide_banner')
    #     .global_args('-loglevel', loglevel)
    #     .run_async(pipe_stdin=True, pipe_stdout=True))

    # cnt = 0
    # for img_lq in img_lqs:
    #     # img_lq = np.clip(img_lq * 255.0, 0, 255)
    #     ffmpeg_img2video.stdin.write(img_lq.astype(np.uint8).tobytes())
    #     cnt += 1

    # ffmpeg_img2video.stdin.close()
    # video_bytes = ffmpeg_img2video.stdout.read()
    # ffmpeg_img2video.wait()
    # # ffmpeg: video to images
    # ffmpeg_video2img.stdin.write(video_bytes)
    # ffmpeg_video2img.stdin.close()
    # img_lqs_ffmpeg = []
    # while True:
    #     in_bytes = ffmpeg_video2img.stdout.read(resized_width * resized_height * 3)
    #     if not in_bytes:
    #         break
    #     in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([resized_height, resized_width, 3]))
    #     in_frame = in_frame.astype(np.float32)
    #     img_lqs_ffmpeg.append(in_frame)

    # ffmpeg_video2img.wait()

    # # save the lq image
    # for img_idx, img_lq in enumerate(img_lqs_ffmpeg):
    #     img_name = os.path.basename(img_list[img_idx])
    #     save_img_lq_path = os.path.join(save_img_path, img_name)
    #     imwrite(img_lq * 255.0, save_img_lq_path)
    # if i == 2:
    #     exit()

os.system(f'rm {tmp_file}')
print('done')