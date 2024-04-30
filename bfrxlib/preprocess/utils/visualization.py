import cv2

def visualize_headpose(img, yaw, pitch, roll):
    from facexlib.visualization.vis_headpose import draw_pose_cube, draw_axis
    font_size = int(img.shape[0] * 0.08)
    axis_size = img.shape[0] * 0.25
    
    yaw_string = (f'y {yaw[0].item():.2f}')
    roll_string = f'r {roll[0].item():.2f}'
    pitch_string = f'p {pitch[0].item():.2f}'

    cv2.putText(img, yaw_string, (font_size, img.shape[0] - font_size), fontFace=1, fontScale=0.5, color=(0, 0, 255), thickness=1)
    cv2.putText(img, roll_string, (font_size, img.shape[0] - font_size*2), fontFace=1, fontScale=0.5, color=(0, 255, 255), thickness=1)
    cv2.putText(img, pitch_string, (font_size, img.shape[0] - font_size*3), fontFace=1, fontScale=0.5, color=(255, 255, 255), thickness=1)
    # draw_pose_cube(img, yaw[0], pitch[0], roll[0], size=size)
    draw_axis(img, yaw[0], pitch[0], roll[0], tdx=50, tdy=50, size=axis_size)


def visualize_5landmark(frame, b):
    '''
    Args:
        frame: frame image to draw
        b: 5 landmarks in shape [1, 10] 
    '''
    # 5 landmarks (for retinaface)
    cv2.circle(frame, (b[0], b[1]), 1, (0, 0, 255), 4)
    cv2.circle(frame, (b[2], b[3]), 1, (0, 255, 255), 4)
    cv2.circle(frame, (b[4], b[5]), 1, (255, 0, 255), 4)
    cv2.circle(frame, (b[6], b[7]), 1, (0, 255, 0), 4)
    cv2.circle(frame, (b[8], b[9]), 1, (255, 0, 0), 4)


def visualize_bbox_rect(frame, d, colors=None):
    '''
    Args:
        d [int]: bbox [x0, y0, x1, y1]
    '''
    if colors is None:
        colors = np.array([[0.1, 0.1, 0.1]])
    # print(colors.shape, colors[d[4] % colors.shape[0], :])
    cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colors[d[4] % colors.shape[0], :] * 255, 3)
    if len(d) > 4:
        cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors[d[4] % colors.shape[0], :] * 255, 2)


def visualize_single_frame(frame, d, face_list, colors, vis_landmark=True, vis_bbox=True):
    if vis_bbox:
        visualize_bbox_rect(frame, d, colors=colors)
    if vis_landmark:
        for b in face_list:
            # confidence
            cv2.putText(frame, f'{b[4]:.4f}', (int(b[0]), int(b[1] + 12)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # bounding boxes
            b = list(map(int, b))
            # cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # 5 landmarks (for retinaface)
            visualize_5landmark(frame, b[5:])

    return frame


def visualize_tracking(frame_list, track_results, save_path, save_clip_size, save_fps=25, vis_landmark=True, vis_bbox=True):
    colors = np.random.rand(32, 3)
    frame_h, frame_w = frame_list[0].shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), save_fps, (frame_w, frame_h))
    for frame_id, (frame, track_res) in enumerate(zip(frame_list, track_results)):
        trackers, face_list = track_res
        for d in trackers:
            d = d.astype(np.int32)
            frame = visualize_single_frame(frame, d, face_list, colors, vis_landmark, vis_bbox)
        writer.write(frame)
    writer.release()
