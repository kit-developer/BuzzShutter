#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import config
from plot import pose_plot as pp
from utils import camera_stream, use_mediapipe, main_process


def stream():

    # カメラ準備
    cap = camera_stream.prepare_camera(config)

    # モデルロード
    # model = use_mediapipe.load_holistic_models(config)
    model = use_mediapipe.load_refine_face_model(config)

    # 位置情報プロット準備
    if config.plot_world_landmark:
        plt, fig, ax = pp.construct()

    photo = np.zeros((400, 600, 3), np.uint8)

    while True:

        # カメラキャプチャ
        ret, image = cap.read()
        if not ret:
            break

        # メイン処理
        debug_image, results, taken = main_process.run(image, model)

        if taken is not None:
            photo = taken
        cv2.imshow("last shot", photo)

        # Pose:位置情報プロット
        if config.plot_world_landmark:
            if results.pose_world_landmarks is not None:
                pp.plot_world_landmarks(plt, ax, results.pose_world_landmarks)

        # キー処理(ESC：終了)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映
        cv2.imshow('BuzzShutter', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def image():

    path = "image_sample/001.png"
    image = cv2.imread(path)

    # モデルロード
    # model = use_mediapipe.load_holistic_models(config)
    model = use_mediapipe.load_refine_face_model(config)

    # メイン処理
    debug_image, results, taken_photo = main_process.run(image, model)

    # 画面反映
    cv2.imshow('BuzzShutter(detection)', debug_image)
    cv2.imshow('BuzzShutter', taken_photo)
    cv2.waitKey(0)


if __name__ == '__main__':
    stream()
