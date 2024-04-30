import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import cv2
from torchvision.transforms.functional import normalize
from facexlib.parsing import init_parsing_model
from facexlib.detection import init_detection_model
from basicsr.utils.img_util import img2tensor, imwrite

face_template_5landmarks = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                        [201.26117, 371.41043], [313.08905, 371.15118]])

parse_map8 = {
    # classes fom parsenet as reference
    0: 0, 14:0, 16:0, 17:0, 18:0, # body & background 
    1: 1, 6:1, 7:1, 8:1, 9:1, # skin
    3: 2, # glasses
    4:3, 5:3, # eyes
    2: 4, # nose
    13: 5, 15: 5, # hair & ear decorations
    11:6, 12:6, # lips
    10: 7, # mouth/teeth
}

def vis_parsing_maps(img, parsing_anno, stride, model_name='parsenet', save_anno_path=None, save_vis_path=None):
    # Colors for all parts
    part_colors = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
                    [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], 
                    [0, 0, 204], # 13: 'hair' red
                    [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    if model_name == 'parsenet':
        pass
        #     0: 'background' 1: 'skin'   2: 'nose'
        #     3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
        #     6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
        #     9: 'r_ear'  10: 'mouth' 11: 'u_lip'
        #     12: 'l_lip' 13: 'hair'  14: 'hat'
        #     15: 'ear_r' 16: 'neck_l'    17: 'neck'
        #     18: 'cloth'
    elif model_name == 'bisenet':
        part_colors = [part_colors[0], part_colors[1], part_colors[6], part_colors[7], part_colors[4],
                       part_colors[5], part_colors[3], part_colors[8], part_colors[9], part_colors[15],
                       part_colors[2], part_colors[10], part_colors[11], part_colors[12], part_colors[17],
                       part_colors[16], part_colors[18], part_colors[13], part_colors[14]]
        # 0: 'background'
        # attributions = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
        #                 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose',
        #                 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l',
        #                 16 'cloth', 17 'hair', 18 'hat']
    else:
        raise NotImplementedError(f'Unknown model name: {model_name}')
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    if save_anno_path is not None:
        if vis_parsing_anno.ndim == 3:
            vis_parsing_anno = vis_parsing_anno[..., 0]
        cv2.imwrite(save_anno_path, vis_parsing_anno)

    if save_vis_path is not None:
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

        vis_im = cv2.addWeighted(img, 0.4, vis_parsing_anno_color, 0.6, 0)

        cv2.imwrite(save_vis_path, vis_im)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--save_path', type=str, required=True)
    parser.add_argument('-res', '--resolution', type=int, default=512)
    parser.add_argument('-m', '--model_name', type=str, default='parsenet', choices=['parsenet', 'bisenet'])
    parser.add_argument('-aligned', '--has_aligned', action='store_true', help='whether the input is aligned')
    parser.add_argument('-class', '--parse_classes', type=int, default=19, choices=[8, 19])
    args = parser.parse_args()

    face_size = (args.resolution, args.resolution)
    device = torch.device('cuda:0')
    model_name = args.model_name
    net = init_parsing_model(model_name=model_name, device=device)
    if not args.has_aligned:
        det_net = init_detection_model('retinaface_resnet50', device=device)

    # save_path = f'/cpfs01/shared/xpixel/dataset/ffhq/ffhq512_images_{model_name}'
    save_path = args.save_path
    save_vis_path = os.path.join(save_path, f'vis_{args.parse_classes}classes')
    save_anno_path = os.path.join(save_path, f'annotation{args.parse_classes}')
    os.makedirs(save_vis_path, exist_ok=True)
    os.makedirs(save_anno_path, exist_ok=True)
    all_ims = sorted(glob.glob(args.input_path+'/*.[jp][pn]g'))

    pbar = tqdm(all_ims)
    for img_path in all_ims:
        pbar.update(1)
        img_basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        pbar.set_description(img_path)
        img_input = img

        # face alignment
        if not args.has_aligned:
            # reference to face restoration helper
            with torch.no_grad():
                # face detection
                bboxes = det_net.detect_faces(img_input, 0.99)
            if len(bboxes) == 0:
                print('Warning: No face detected in %s' % img_path)
                continue
            bbox = bboxes[0]
            landmark = np.array([[bbox[i], bbox[i + 1]] for i in range(5, 15, 2)])
            affine_matrix = cv2.estimateAffinePartial2D(
                landmark, face_template_5landmarks, method=cv2.LMEDS)[0]
            inverse_affine = cv2.invertAffineTransform(affine_matrix)

            cropped_face = cv2.warpAffine(
                img_input, affine_matrix, face_size, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=(135, 133, 132))  # gray
            img_input = cropped_face

        # inference for parsing
        # resize to 512 x 512 for better performance
        if img_input.shape[0] != face_size[0]:
            img_input = cv2.resize(img_input, face_size, interpolation=cv2.INTER_LINEAR)
        img_t = img2tensor(img_input.astype('float32') / 255., bgr2rgb=True, float32=True)
        if model_name == 'parsenet':
            normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        elif model_name == 'bisenet':
            normalize(img_t, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        img_t = torch.unsqueeze(img_t, 0).to(device)

        with torch.no_grad():
            out = net(img_t)[0]
        # print(out.squeeze().shape)
        out = out.squeeze(0).cpu().numpy().argmax(0)

        # print(out.shape, out)
        if args.parse_classes == 8:
            out_new = np.zeros_like(out)
            for i in range(len(parse_map8.keys())):
                out_new[out == i] = parse_map8[i]
            out = out_new

        # warp parsemap back if the input is not aligned
        if not args.has_aligned:
            out = cv2.warpAffine(
                out.astype(np.uint8), inverse_affine, face_size, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0)
        # vis_parsing_maps(
        #     img_input,
        #     out,
        #     model_name=model_name,
        #     stride=1)
        vis_parsing_maps(
            img,
            out,
            model_name=model_name,
            stride=1,
            save_vis_path=os.path.join(save_vis_path, f'{img_basename}'),
            save_anno_path=os.path.join(save_anno_path, f'{img_basename[:-4]}.png'))
        # exit()