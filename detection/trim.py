import numpy as np
from detection import keypoints as kp


def trim_eye(image, results, config):

    height = 10
    width = 30
    eye_images = [np.zeros((height, width, 3)), np.zeros((height, width, 3))]

    if results.face_landmarks is not None:
        face_pts = get_landmarks_points(image, results.face_landmarks)

        # 目の上側と下側を合わせたidxリストを準備
        left_eye_idx_list = kp.left_eye_upper_idx
        left_eye_idx_list.extend(kp.left_eye_lower_idx)
        right_eye_idx_list = kp.right_eye_upper_idx
        right_eye_idx_list.extend(kp.right_eye_lower_idx)

        # 目を抜き出す座標を左右それぞれ取得
        boxes = []
        eye_images = []
        for idx_list in [left_eye_idx_list, right_eye_idx_list]:
            min_x, max_x, min_y, max_y = get_range(image, face_pts, idx_list)

            eye_images.append(image[min_y:max_y, min_x:max_x])
            # boxes.append(get_corner(min_x, max_x, min_y, max_y))

    return eye_images


def get_range(image, landmarks_points, idx_list):
    min_x = image.shape[1]
    max_x = 0
    min_y = image.shape[0]
    max_y = 0
    for idx in idx_list:
        min_x = min(landmarks_points[idx][0], min_x)
        max_x = max(landmarks_points[idx][0], max_x)
        min_y = min(landmarks_points[idx][1], min_y)
        max_y = max(landmarks_points[idx][1], max_y)
    return min_x, max_x, min_y, max_y


def get_corner(left, right, lower, upper):
    upper_left = [left, upper]
    upper_right = [right, upper]
    lower_left = [left, lower]
    lower_right = [right, lower]
    return np.array([upper_left, upper_right, lower_left, lower_right])


def get_landmarks_points(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    if landmarks is not None:
        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_z = landmark.z

            landmark_point.append((landmark_x, landmark_y))

    return landmark_point
