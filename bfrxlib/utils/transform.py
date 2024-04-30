import cv2
import numpy as np
from skimage import transform as trans


def landmark2D(landmarks5):
    return np.array([[landmarks5[i], landmarks5[i + 1]] for i in range(0, 10, 2)])

def align_face(cv_img, outsize, target_face, landmarks='ffhq'):
    """Align input face based on different standard face templates.
    params:
        target_face: Option: ['mtcnn', 'ffhq']
        outsize: ouput size in tuple. e.g. (112, 112).
        landmarks (str or np.array): 5 landmarks in input cv_img. 
            Default: 'ffhq', means cv_img is a face image aligned to ffhq face.

    1) align_face(img, (112, 112), target_face='mtcnn', landmarks='ffhq'): to transform an input ffhq face to an mtcnn face.
    2)  landmarks = np.array([[22, 51], [64, 50], [40, 70], [31, 90], [62, 90]]) # 5 landmarks in img
        align_face(img, (112, 112), target_face='mtcnn', landmarks=landmarks): to transform an input random face to an mtcnn face.

    Most FIQA models were trained in face imaged aligned with the mtcnn-face.
    Deviation of the face alignment can result in non-trivial drop on the evaluation score.
    """
    face_templates = {
        # Format of 5-landmark templates: [[x1, y1], ..., [x5, y5]]
        # standard face 5 landmarks (image size 112 x 96/112) for face-related models based on MTCNN (arcface).
        # shift (112 × 96) -> (112 × 112) a width difference: (112 - 96) / 2
        'mtcnn': (np.array([[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
                           [33.5493, 92.3655], [62.7299, 92.2041]]) + (8., 0)) / (112, 112),
        # standard 5 landmarks for FFHQ faces with 512 x 512 
        'ffhq': np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                          [201.26117, 371.41043], [313.08905, 371.15118]]) / (512, 512)
    }
    assert target_face in face_templates, f'Unsupported face type {target_face}!'
    target_landmarks = face_templates[target_face].astype('float32') * outsize

    h, w = cv_img.shape[:2]
    landmarks = face_templates[landmarks] * (w, h) if isinstance(landmarks, str) else landmarks
    landmarks = landmark2D(landmarks) if landmarks.ndim == 1 else landmarks
    assert landmarks.shape == (5, 2), f'Error! Illegal landmarks shape {landmarks.shape}. Expect input landmarks with a shape of 1D(10,) or 2D(5, 2).'

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, target_landmarks)
    M = tform.params[:2,:]
    aligned_face = cv2.warpAffine(cv_img, M, outsize, borderValue=0.0)

    return aligned_face