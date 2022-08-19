import mediapipe as mp
import cv2


def load_holistic_models(config):

    smooth_landmarks = not config.unuse_smooth_landmarks
    model_complexity = config.model_complexity
    min_detection_confidence = config.min_detection_confidence
    min_tracking_confidence = config.min_tracking_confidence

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

    return holistic


def load_refine_face_model(config):

    min_detection_confidence = config.min_detection_confidence
    min_tracking_confidence = config.min_tracking_confidence
    refine_landmarks = config.refine_landmarks

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_model = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        refine_landmarks=refine_landmarks,
    )

    return face_mesh_model


def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results
