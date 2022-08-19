import numpy as np

from draw import draw_face_parts as dfp
from detection import keypoints as kp


def iris_position(image, results):

    iris_rates = []
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:

            # 推論結果から座標情報を取得
            landmark_points = dfp.get_landmark_points(image, face_landmarks)

            # 目頭と目尻を結ぶ線を中心線とする
            l_line = pts2line(landmark_points[kp.left_eye_inner_corner], landmark_points[kp.left_eye_outer_corner])
            r_line = pts2line(landmark_points[kp.right_eye_inner_corner], landmark_points[kp.right_eye_outer_corner])

            # 中心線と、最も離れている　上　まぶたの距離を計る
            l_u_max_dist = 0
            for idx in kp.left_eye_upper_idx:
                l_u_max_dist = max(distance_pts_line(landmark_points[idx], l_line), l_u_max_dist)
            r_u_max_dist = 0
            for idx in kp.right_eye_upper_idx:
                r_u_max_dist = max(distance_pts_line(landmark_points[idx], r_line), r_u_max_dist)

            # 中心線と、最も離れている　下　まぶたの距離を計る
            l_l_max_dist = 0
            for idx in kp.left_eye_lower_idx:
                l_l_max_dist = max(distance_pts_line(landmark_points[idx], l_line), l_l_max_dist)
            r_l_max_dist = 0
            for idx in kp.right_eye_lower_idx:
                r_l_max_dist = max(distance_pts_line(landmark_points[idx], r_line), r_l_max_dist)

            # 虹彩の中心と中心線との距離を計る
            l_iris_dist = distance_pts_line(landmark_points[kp.left_iris_center], l_line)
            r_iris_dist = distance_pts_line(landmark_points[kp.right_iris_center], r_line)

            # 虹彩がどの位置（割合）にあるか調べる
            l_iris_rate, r_iris_rate = iris_position_rate(l_iris_dist, l_u_max_dist, l_l_max_dist,
                                                          r_iris_dist, r_u_max_dist, r_l_max_dist)

            # 眉間２点の距離（画像に占める顔の大きさの標準化に利用）
            pt1 = np.array(landmark_points[kp.between_the_eyebrows[0]])
            pt2 = np.array(landmark_points[kp.between_the_eyebrows[1]])
            dist = np.linalg.norm(pt1 - pt2)

            # まぶたの開度
            l_open = (l_u_max_dist + l_l_max_dist) / dist
            r_open = (r_u_max_dist + r_l_max_dist) / dist

            # print("左目", round(l_iris_rate, 2), round(l_iris_dist, 2), round(l_u_max_dist, 2), round(l_l_max_dist, 2),
            #       "      右目", round(r_iris_rate, 2), round(r_iris_dist, 2), round(r_u_max_dist, 2), round(r_l_max_dist, 2))

            # print("左目", l_open, "   右目", r_open)

            # rates = [l_iris_rate, r_iris_rate]
            rates = [l_iris_rate * l_open, r_iris_rate * r_open]

            iris_rates.append(rates)

    return iris_rates


def iris_position_rate(l_iris_dist, l_u_max_dist, l_l_max_dist,
                       r_iris_dist, r_u_max_dist, r_l_max_dist):

    l_u_iris_rate = l_iris_dist / l_u_max_dist
    r_u_iris_rate = r_iris_dist / r_u_max_dist
    l_l_iris_rate = l_iris_dist / l_l_max_dist
    r_l_iris_rate = r_iris_dist / r_l_max_dist

    if l_u_iris_rate < l_l_iris_rate:
        l_iris_rate = l_u_iris_rate
    else:
        l_iris_rate = - l_l_iris_rate

    if r_u_iris_rate < r_l_iris_rate:
        r_iris_rate = r_u_iris_rate
    else:
        r_iris_rate = - r_l_iris_rate

    return l_iris_rate, r_iris_rate


# ２点を通る直線の方程式を求める
def pts2line(p1, p2):
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p1[1] * p2[0] - p1[0] * p2[1]
    return a, b, c


# 点と直線の距離
def distance_pts_line(pt, line):

    x, y = pt
    a, b, c = line
    distance = abs(a*x + b*y + c)/np.sqrt(a**2 + b**2)
    return distance
