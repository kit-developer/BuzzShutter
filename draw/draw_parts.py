import numpy as np
import cv2
import copy

from draw import draw_points as dp
from draw import draw_face_parts as dfp


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

    return debug_image


# def iris(image, face_mesh_model, iris_detector, debug_image):
#     # Face Mesh検出
#     face_results = face_mesh_model(image)
#     for face_result in face_results:
#         # 目周辺のバウンディングボックス計算
#         left_eye, right_eye = face_mesh_model.calc_around_eye_bbox(face_result)
#
#         # 虹彩検出
#         left_iris, right_iris = detect_iris(image, iris_detector, left_eye,
#                                             right_eye)
#
#         # 虹彩の外接円を計算
#         left_center, left_radius = calc_min_enc_losingCircle(left_iris)
#         right_center, right_radius = calc_min_enc_losingCircle(right_iris)
#
#         # デバッグ描画
#         debug_image = dp.draw_iris_landmark(
#             debug_image,
#             left_iris,
#             right_iris,
#             left_center,
#             left_radius,
#             right_center,
#             right_radius,
#         )
#
#
# def detect_iris(image, iris_detector, left_eye, right_eye):
#     image_width, image_height = image.shape[1], image.shape[0]
#     input_shape = iris_detector.get_input_shape()
#
#     # 左目
#     # 目の周辺の画像を切り抜き
#     left_eye_x1 = max(left_eye[0], 0)
#     left_eye_y1 = max(left_eye[1], 0)
#     left_eye_x2 = min(left_eye[2], image_width)
#     left_eye_y2 = min(left_eye[3], image_height)
#     left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
#                                          left_eye_x1:left_eye_x2])
#     # 虹彩検出
#     eye_contour, iris = iris_detector(left_eye_image)
#     # 座標を相対座標から絶対座標に変換
#     left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)
#
#     # 右目
#     # 目の周辺の画像を切り抜き
#     right_eye_x1 = max(right_eye[0], 0)
#     right_eye_y1 = max(right_eye[1], 0)
#     right_eye_x2 = min(right_eye[2], image_width)
#     right_eye_y2 = min(right_eye[3], image_height)
#     right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
#                                           right_eye_x1:right_eye_x2])
#     # 虹彩検出
#     eye_contour, iris = iris_detector(right_eye_image)
#     # 座標を相対座標から絶対座標に変換
#     right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)
#
#     return left_iris, right_iris
#
#
# def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
#     iris_list = []
#     for index in range(5):
#         point_x = int(iris[index * 3] *
#                       ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
#         point_y = int(iris[index * 3 + 1] *
#                       ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
#         point_x += eye_bbox[0]
#         point_y += eye_bbox[1]
#
#         iris_list.append((point_x, point_y))
#
#     return iris_list
#
#
# def calc_min_enc_losingCircle(landmark_list):
#     center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
#     center = (int(center[0]), int(center[1]))
#     radius = int(radius)
#
#     return center, radius


def face_mesh_neo(debug_image, results, config):
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            # 外接矩形の計算
            brect = calc_bounding_rect(debug_image, face_landmarks)
            # 虹彩の外接円の計算
            left_eye, right_eye = None, None
            if config.refine_landmarks:
                left_eye, right_eye = dfp.calc_iris_min_enc_losingCircle(
                    debug_image,
                    face_landmarks,
                )
            # 描画
            debug_image = dfp.draw_landmarks(
                debug_image,
                face_landmarks,
                config.refine_landmarks,
                left_eye,
                right_eye,
            )
            debug_image = dfp.draw_bounding_rect(config.use_brect, debug_image, brect)

    return debug_image









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
