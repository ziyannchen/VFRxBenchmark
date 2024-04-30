

if __name__ == '__main__':
    from bfrxlib.preprocess.utils import read_pos_info, get_landmarks2D, DBmap, makedir
    from bfrxlib.utils.face_restoration_helper import FaceRestoreHelper
    from basicsr.utils import scandir

    import glob
    import os
    import cv2
    from pathlib import Path
    import numpy as np
    import argparse
    import torch
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-s', '--resize', type=int, default=512, help='Resize the input image to this size.')

    # align from the face detector
    parser.add_argument('-w', '--write_landmarks_dir', type=str, default=None, help='Write landmarks to this directory. Format: [b1, b2, b3, b4, [x1, y1], [x2, y2], ...]')
    # parser.add_argument('-l', '--only_keep_largest', action='store_true', help='Only keep the largest face in the image.')
    parser.add_argument('-l', '--only_keep_largest', type=bool, default=True, help='Only keep the largest face in the image.')

    # align from pre-defined pos dir
    parser.add_argument('-p', '--pos_dir', type=str, default=None, help='To affine the face from pre-defined position info file.')
    parser.add_argument('-scale', '--target_scale', type=int, default=1, 
        help='Downsample the landmarks in pos_dir to match the resized target image inputs. \
              e.g. landmarks pos_dir is for 512*512, when resized target image is 512, scale=1; when resized target image is 128, scale=4.')
    args = parser.parse_args()

    if args.pos_dir is not None:
        print('No directory of position info given. Using detector to detect faces.')

    # proj_dir = Path(__file__).resolve().parents[2]
    if args.output_dir is None:
        output_dir = args.input_dir+'_aligned'
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.write_landmarks_dir is not None:
        os.makedirs(args.write_landmarks_dir, exist_ok=True)
    
    pos_dir = args.pos_dir

    all_inputs = sorted(os.listdir(args.input_dir))
    # all_inputs = sorted(scandir(args.input_dir, recursive=True))
    print(all_inputs)
    print('Input data length: ', len(all_inputs))

    if len(all_inputs) and os.path.isdir(os.path.join(args.input_dir, all_inputs[0])):
        # video frame dirs
        is_frames_dir = True
    else:
        # image dir
        is_frames_dir = False
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_helper = FaceRestoreHelper(
        1,
        face_size=args.resize,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device)

    # prefix = 'final_results'
    prefix = ''
    pbar = tqdm(total=len(all_inputs))
    for basename in all_inputs:
        pbar.update(1)
        pbar.set_description(basename)
        landmarks_save = []
        if is_frames_dir:
            # clip/frames dir
            clip_basename = basename
            frames_dirname = os.path.join(args.input_dir, clip_basename)
            save_dir = makedir((output_dir, clip_basename))
            if pos_dir is not None:
                pos_info = read_pos_info(clip_basename, pos_dir)
            
            all_frames = sorted(glob.glob(os.path.join(frames_dirname, prefix, '*.png')))
            for ind, frame_path in enumerate(all_frames):
                frame = cv2.imread(frame_path)
                img_name = os.path.basename(frame_path)
                face_helper.read_image(frame)

                if pos_dir is not None:
                    _, landmark = get_landmarks2D(img_name, pos_info)
                    landmark /= args.target_scale
                    face_helper.all_landmarks_5 = [landmark]
                else:
                    # clip: only keep the largest face to keep the consistency
                    face_helper.get_face_landmarks_5(only_keep_largest=True)
                landmarks_save.append([face_helper.det_faces, face_helper.all_landmarks_5])
                # print('Saving to ', os.path.join(save_dir, img_name))
                face_helper.align_warp_face()
                save_cropped_path = os.path.join(save_dir, prefix, img_name)

                if len(face_helper.cropped_faces):
                    target_face = face_helper.cropped_faces[0]
                    # if target_face.shape[0] != args.resize:
                    #     target_face = cv2.resize(target_face, (args.resize, args.resize), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(save_cropped_path, target_face, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    print(f'Warning: No face detected in ', frame_path, f'Skipping clip {frames_dirname}...')
                    break
                face_helper.clean_all()

        else:
            # image dir
            input_path = os.path.join(args.input_dir, basename)
            frame = cv2.imread(input_path)
            # print(frame)
            face_helper.read_image(frame)

            if pos_dir is not None:
                _, landmark = get_landmarks2D(basename, pos_info)
                face_helper.all_landmarks_5 = [landmark]
            else:
                face_helper.get_face_landmarks_5(only_keep_largest=args.only_keep_largest)
                landmarks_save = [face_helper.det_faces, face_helper.all_landmarks_5]
                
            # print('Saving to ', os.path.join(save_dir, img_name))
            face_helper.align_warp_face()

            save_cropped_path = os.path.join(output_dir, prefix, basename)
            if len(face_helper.cropped_faces) > 0:
                target_face = face_helper.cropped_faces[0]
                if target_face.shape[0] != args.resize:
                    target_face = cv2.resize(target_face, (args.resize, args.resize), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(save_cropped_path, target_face, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                print(f'Warning: No face detected in ', input_path, 'Skipping...')
                break
            face_helper.clean_all()

        if args.write_landmarks_dir:
            with open(os.path.join(args.write_landmarks_dir, basename+'.txt'), 'w') as f:
                for det_face, landmarks5 in landmarks_save:
                    line_str = ' '.join([str(i) for i in det_face[:4]]) + ' '
                    line_str += ' '.join([str(i) for i in np.array(landmarks5).reshape(-1)])
                    f.write(line_str+'\n')