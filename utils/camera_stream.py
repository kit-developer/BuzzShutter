import cv2


def prepare_camera(config):
    cap = cv2.VideoCapture(config.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    return cap