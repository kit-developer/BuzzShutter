#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2
import mediapipe as mp

import config
from draw import draw_parts


def main():

    cap_device = config.device
    cap_width = config.width
    cap_height = config.height

    # upper_body_only = config.upper_body_only
    smooth_landmarks = not config.unuse_smooth_landmarks
    model_complexity = config.model_complexity
    min_detection_confidence = config.min_detection_confidence
    min_tracking_confidence = config.min_tracking_confidence
    refine_landmarks = config.refine_landmarks

    # カメラ準備
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        # enable_segmentation=enable_segmentation,    ###
        # smooth_segmentation=smooth_segmentation,    ###
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_model = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        refine_landmarks=refine_landmarks,
    )

    # 位置情報プロット準備
    if config.plot_world_landmark:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    while True:

        # カメラキャプチャ
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_holistic = holistic.process(image)
        results_face = face_mesh_model.process(image)
        image.flags.writeable = True

        # Face Mesh
        # draw_parts.face_mesh(debug_image, results_holistic, config)

        # Pose
        # draw_parts.pose(debug_image, results, config)

        # Pose:位置情報プロット
        if config.plot_world_landmark:
            if results_holistic.pose_world_landmarks is not None:
                plot_world_landmarks(plt, ax, results_holistic.pose_world_landmarks)

        # Hands
        # draw_parts.left_hand(debug_image, results, config)
        # draw_parts.right_hand(debug_image, results, config)

        # 目を切り抜き
        debug_image = draw_parts.face_mesh_neo(debug_image, results_face, config)

        # キー処理(ESC：終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映
        cv2.imshow('BuzzShutter', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def plot_world_landmarks(
        plt,
        ax,
        landmarks,
        visibility_th=0.5,
):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))

    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)

    plt.pause(.001)

    return


if __name__ == '__main__':
    main()