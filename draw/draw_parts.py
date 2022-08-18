import numpy as np
import cv2

from draw import draw_points as dp


def face_mesh(debug_image, results, config):
    face_landmarks = results.face_landmarks
    if face_landmarks is not None:
        # 外接矩形の計算
        brect = calc_bounding_rect(debug_image, face_landmarks)
        # 描画
        debug_image = dp.draw_face_landmarks(debug_image, face_landmarks)
        debug_image = dp.draw_bounding_rect(config.use_brect, debug_image, brect)

    return debug_image


def pose(debug_image, results, config):

    if config.enable_segmentation and results.segmentation_mask is not None:
        # セグメンテーション
        mask = np.stack((results.segmentation_mask,) * 3,
                        axis=-1) > config.segmentation_score_th
        bg_resize_image = np.zeros(debug_image.shape, dtype=np.uint8)   # image -> debug_image
        bg_resize_image[:] = (0, 255, 0)
        debug_image = np.where(mask, debug_image, bg_resize_image)

    pose_landmarks = results.pose_landmarks
    if pose_landmarks is not None:
        # 外接矩形の計算
        brect = calc_bounding_rect(debug_image, pose_landmarks)
        # 描画
        debug_image = dp.draw_pose_landmarks(
            debug_image,
            pose_landmarks,
            # upper_body_only,
        )
        debug_image = dp.draw_bounding_rect(config.use_brect, debug_image, brect)

    return debug_image


def left_hand(debug_image, results, config):
    left_hand_landmarks = results.left_hand_landmarks
    if left_hand_landmarks is not None:
        # 手の平重心計算
        cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
        # 外接矩形の計算
        brect = calc_bounding_rect(debug_image, left_hand_landmarks)
        # 描画
        debug_image = dp.draw_hands_landmarks(
            debug_image,
            cx,
            cy,
            left_hand_landmarks,
            # upper_body_only,
            'R',
        )
        debug_image = dp.draw_bounding_rect(config.use_brect, debug_image, brect)

    return debug_image


def right_hand(debug_image, results, config):
    right_hand_landmarks = results.right_hand_landmarks
    if right_hand_landmarks is not None:
        # 手の平重心計算
        cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
        # 外接矩形の計算
        brect = calc_bounding_rect(debug_image, right_hand_landmarks)
        # 描画
        debug_image = dp.draw_hands_landmarks(
            debug_image,
            cx,
            cy,
            right_hand_landmarks,
            # upper_body_only,
            'L',
        )
        debug_image = dp.draw_bounding_rect(config.use_brect, debug_image, brect)










def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]
