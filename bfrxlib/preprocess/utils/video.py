import glob
import os
import cv2
import ffmpeg
from natsort import natsorted

class ReadError(Exception):
    # define a read exception with image
    def __str__(self):
        return 'Unsuccessfully read frames.'

def load_frames(frames_path, name_list=False):
    '''
    Load sequence of frames from a folder as numpy array list.
    Args:
        frames_path: the folder path of frames.
    '''
    frame_paths = natsorted(glob.glob(os.path.join(frames_path, '*.[jp][pn]g')))
    print(os.path.join(frames_path, '*.[jp][pn]g'), len(frame_paths))
    frame_list = []
    for path in frame_paths:
        frame = cv2.imread(path)
        frame_list.append(frame)
    if name_list:
        return frame_list, frame_paths
    return frame_list


def split_list_by_n(list_collection, n):
    for i in range(0, len(list_collection), n):
        yield list_collection[i: i + n]


def images2video(img_source, save_path, clip_size=None, fps=25, format="mp4v", use_ffmpeg=False, suffix='png', **extra_args):
    """_summary_

    Args:
        img_source (_type_): the frames folder path(str) or a frame list(list).
        save_path (_type_): video save path
        clip_size (_type_, optional): clip frame width(height). Defaults to None.
        fps (int, optional): frames per second. Defaults to 25.
        format (str, optional): for cv2.VideoWriter. Options: ['mp4v', 'H264']. Defaults to "mp4v".
            p.s. 'h264' is the format of MP4 = MPEG 4 which is supported by player of browsers.
        ffmpeg: use ffmpeg to write video
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # FIXME: ffmpeg not standard: e.g. resize etc.
    if use_ffmpeg:
        assert format in ['h264'], 'ffmpeg only support h264 format.'
        vcodec = 'libx264'
        if isinstance(img_source, str):
            # convert frames in a folder into a video using ffmpeg
            # assert isinstance(img_source, str), 'ffmpeg only support frames folder as input.'
            cmd = f'ffmpeg -framerate {fps} -pattern_type glob -i \'{img_source}/*.{suffix}\' -pix_fmt yuv420p -c:v {vcodec} {save_path}'
            print(cmd)
            os.system(cmd)
        else:
            raise TypeError('ffmpeg only support frames folder as input.')
        # elif isinstance(img_source, list):
            # h, w = img_source[0].shape[:2]
            # ffmpeg_img2video = (
            #     ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps, s=f'{w}x{h}',)
            #           .filter('fps', fps=fps, round='up')
            #           .output('pipe:', format=format, pix_fmt='yuv420p', vcodec=vcodec, **extra_args)
            #           .global_args('-hide_banner')
            #           .run_async(pipe_stdin=True, pipe_stdout=True))
            # print('000000000')
            # # read a sequence of images
            # for idx, img in enumerate(img_source):
            #     print(1111, idx, img.shape)
            #     ffmpeg_img2video.stdin.write(img.tobytes())
            # print('hahhhah')
            # ffmpeg_img2video.stdin.close()
            # ffmpeg.output(ffmpeg_img2video, save_path).run()
        return

    if isinstance(img_source, str):
        frame_list = load_frames(img_source)
    elif isinstance(img_source, list):
        frame_list = natsorted(img_source)
    else:
        raise TypeError('img_source should be str or list.')

    if clip_size is None:
        frame_h, frame_w = frame_list[0].shape[:2]
        clip_size = (frame_w, frame_h)

    # handle the issue that cv2 video writer cannot write video with h264 format
    cv2_format = {'mp4v': 'mp4v', 'h264': 'mp4v'}
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*cv2_format[format]), fps, clip_size)

    for frame in frame_list:
        if frame.shape[:2] != clip_size:
            frame = cv2.resize(frame, clip_size, interpolation=cv2.INTER_CUBIC)
        writer.write(frame)
    writer.release()

    if format == 'h264':
        # convert to mp4
        cmd = f'ffmpeg -i {save_path} -c:v libx264 -c:a copy {save_path[:-4]}_h264.mp4'
        os.system(cmd)

def video2images(video_path, interval=1, rgb=False):
    '''
    Read frames from video using cv2.
    return frames list.
    '''
    frame_list = []
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    if not success:
        raise ReadError()
    
    while success:
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame)
        success, frame = vidcap.read()
    return frame_list[::interval]