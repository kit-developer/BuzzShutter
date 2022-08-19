import copy
import cv2
import numpy as np

import config
from draw import draw_parts
from utils import use_mediapipe
from detection import calc_eye_width as cew


def run(image, model):

    taked_photo = None

    image = cv2.flip(image, 1)  # ミラー表示
    debug_image = copy.deepcopy(image)

    # 検出実施
    results = use_mediapipe.detection(image, model)

    # Face Mesh
    # draw_parts.face_mesh(debug_image, results, config)

    # Face Mesh (with Eye Track) -> require load_refine_face_model()
    debug_image = draw_parts.face_mesh_neo(debug_image, results, config)

    iris_position = cew.iris_position(debug_image, results)

    if len(iris_position) > 0:
        print(sum(iris_position[0]))
        if sum(iris_position[0]) > 0.8:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
            taken_photo = image

    # Pose
    # draw_parts.pose(debug_image, results, config)

    # Hands
    # draw_parts.left_hand(debug_image, results, config)
    # draw_parts.right_hand(debug_image, results, config)

    return debug_image, results, taken_photo
