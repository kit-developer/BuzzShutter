import numpy as np
import cv2
from draw import draw_face_parts as dfp
from detection import keypoints as kp


#sepia effect
def sepia(img):
    img_sepia = np.array(img, dtype=np.float64) # converting to float to prevent loss
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                    [0.349, 0.686, 0.168],
                                    [0.393, 0.769, 0.189]])) # multipying image with special sepia matrix
    img_sepia[np.where(img_sepia > 255)] = 255 # normalizing values greater than 255 to 255
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    return img_sepia


#grey pencil sketch effect
def pencil_sketch_grey(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_gray


#colour pencil sketch effect
def pencil_sketch_col(img):
    #inbuilt function to create sketch effect in colour and greyscale
    sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
    return sk_color


#HDR effect
def HDR(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return  hdr


#summer effect
def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum


#winter effect
def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win


#defining a function
from scipy.interpolate import UnivariateSpline
def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))


nekomimi_image = cv2.imread("image_sample/nekomimi_black1.png")
nekohige_image = cv2.imread("image_sample/nekohige2.png")


def overlay_filter(results, image):

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            land_pts = dfp.get_landmark_points(image, face_landmarks)

            # 猫耳
            left_lower, right_lower = land_pts[kp.right_head], land_pts[kp.left_head]
            src_pts = [[0, 0], [nekomimi_image.shape[1], 0], [0, nekomimi_image.shape[0]], [nekomimi_image.shape[1], nekomimi_image.shape[0]]]
            h_w = nekomimi_image.shape[0] / nekomimi_image.shape[1]
            theta = np.arctan2(right_lower[1] - left_lower[1], right_lower[0] - left_lower[0]) + np.pi
            w = np.linalg.norm(np.array(left_lower) - np.array(right_lower))
            left_upper = [- w * h_w * np.sin(theta) + left_lower[0], w * h_w * np.cos(theta) + left_lower[1]]
            right_upper = [- w * h_w * np.sin(theta) + right_lower[0], w * h_w * np.cos(theta) + right_lower[1]]

            dst_pts = [left_upper, right_upper, left_lower, right_lower]
            overlay_image = overlay(nekomimi_image, src_pts, image, dst_pts)

            # 猫ひげ
            left, right, center = land_pts[kp.left_face], land_pts[kp.right_face], land_pts[kp.nose_top]
            if np.linalg.norm(np.array(left) - np.array(center)) > np.linalg.norm(np.array(right) - np.array(center)):

                # 左と鼻を利用
                src_pts = [[0, 0], [nekohige_image.shape[1]/2, 0], [0, nekohige_image.shape[0]],
                           [nekohige_image.shape[1]/2, nekohige_image.shape[0]]]
                h_w = nekohige_image.shape[1] / (nekohige_image.shape[0]/2)
                theta = np.arctan2(center[1] - left[1], center[0] - left[0]) + np.pi
                w = np.linalg.norm(np.array(left) - np.array(center))
                dst_pts = [[- w * h_w / 2 * np.sin(theta) + left[0], w * h_w / 2 * np.cos(theta) + left[1]],
                           [- w * h_w / 2 * np.sin(theta) + center[0], w * h_w / 2 * np.cos(theta) + center[1]],
                           [w * h_w / 2 * np.sin(theta) + left[0], - w * h_w / 2 * np.cos(theta) + left[1]],
                           [w * h_w / 2 * np.sin(theta) + center[0], - w * h_w / 2 * np.cos(theta) + center[1]]]

            else:

                # 右と鼻を利用
                src_pts = [[nekohige_image.shape[1]/2, 0], [nekohige_image.shape[1], 0],
                           [nekohige_image.shape[1]/2, nekohige_image.shape[0]], [nekohige_image.shape[1], nekohige_image.shape[0]]]
                h_w = nekohige_image.shape[1] / (nekohige_image.shape[0]/2)
                theta = np.arctan2(right[1] - center[1], right[0] - center[0]) + np.pi
                w = np.linalg.norm(np.array(center) - np.array(right))
                dst_pts = [[- w * h_w / 2 * np.sin(theta) + center[0], w * h_w / 2 * np.cos(theta) + center[1]],
                           [- w * h_w / 2 * np.sin(theta) + right[0], w * h_w / 2 * np.cos(theta) + right[1]],
                           [w * h_w / 2 * np.sin(theta) + center[0], - w * h_w / 2 * np.cos(theta) + center[1]],
                           [w * h_w / 2 * np.sin(theta) + right[0], - w * h_w / 2 * np.cos(theta) + right[1]]]

            overlay_image = overlay(nekohige_image, src_pts, overlay_image, dst_pts)

            return overlay_image


def overlay(fore_image, src_pts, back_image, dst_pts):

    mask = np.zeros(back_image.shape, np.uint8)
    mask[0:fore_image.shape[0], 0:fore_image.shape[1]] = fore_image

    # left_upper, right_upper, left_lower, right_lower
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    perspective_mask = cv2.warpPerspective(mask, mat, back_image.shape[1::-1])
    overlay_image = np.where(perspective_mask[:, :, :] == 0, back_image, perspective_mask)

    return overlay_image

