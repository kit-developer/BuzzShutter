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


def overlay_filter(results, image):
    fore_image = cv2.imread("image_sample/nekomimi_black1.png")

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:

            # 猫耳
            land_pts = dfp.get_landmark_points(image, face_landmarks)
            left_lower, right_lower = land_pts[kp.right_head], land_pts[kp.left_head]

            h_w = fore_image.shape[0] / fore_image.shape[1]
            theta = np.arctan2(fore_image.shape[0], fore_image.shape[1])
            w = np.linalg.norm(np.array(left_lower) - np.array(right_lower))
            left_upper = [w * h_w * np.sin(theta) + left_lower[0], - w * h_w * np.cos(theta) + left_lower[1]]
            right_upper = [w * h_w * np.sin(theta) + right_lower[0], - w * h_w * np.cos(theta) + right_lower[1]]
            pts = [left_upper, right_upper, left_lower, right_lower]

            nekomimi_overlay = overlay(fore_image, image, pts)

            return nekomimi_overlay


def overlay(fore_image, back_image, pts):

    mask = np.zeros(back_image.shape, np.uint8)
    mask[0:fore_image.shape[0], 0:fore_image.shape[1]] = fore_image

    src_pts = np.array([[0, 0],
                        [fore_image.shape[1], 0],
                        [0, fore_image.shape[0]],
                        [fore_image.shape[1], fore_image.shape[0]]],
                       dtype=np.float32)

    # left_upper, right_upper, left_lower, right_lower
    dst_pts = np.array(pts, dtype=np.float32)

    mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    perspective_mask = cv2.warpPerspective(mask, mat, back_image.shape[1::-1])
    overlay_image = np.where(perspective_mask[:, :, :] == 0, back_image, perspective_mask)

    return overlay_image

