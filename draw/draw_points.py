import cv2


def draw_iris_landmark(
    debug_image,
    left_iris,
    right_iris,
    left_center,
    left_radius,
    right_center,
    right_radius,
):
    # 虹彩：外接円
    cv2.circle(debug_image, left_center, left_radius, (0, 255, 0), 2)
    cv2.circle(debug_image, right_center, right_radius, (0, 255, 0), 2)

    # 虹彩：ランドマーク
    for point in left_iris:
        cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)
    for point in right_iris:
        cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 2)

    # 虹彩：半径
    cv2.putText(debug_image, 'r:' + str(left_radius) + 'px',
               (left_center[0] + int(left_radius * 1.5),
                left_center[1] + int(left_radius * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.putText(debug_image, 'r:' + str(right_radius) + 'px',
               (right_center[0] + int(right_radius * 1.5),
                right_center[1] + int(right_radius * 0.5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return debug_image




def draw_hands_landmarks(
        image,
        cx,
        cy,
        landmarks,
        # upper_body_only,
        handedness_str='R'):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv2.LINE_AA)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv2.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv2.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv2.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv2.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv2.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv2.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv2.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv2.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv2.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv2.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv2.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv2.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv2.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv2.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv2.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv2.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv2.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv2.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv2.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        cv2.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        cv2.putText(image, handedness_str, (cx - 6, cy + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return image


def draw_face_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        cv2.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # 左眉毛(55：内側、46：外側)
        cv2.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv2.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv2.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv2.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv2.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # 左目 (133：目頭、246：目尻)
        cv2.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv2.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # 右目 (362：目頭、466：目尻)
        cv2.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv2.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # 口 (308：右端、78：左端)
        cv2.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv2.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv2.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv2.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv2.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv2.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv2.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv2.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv2.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv2.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv2.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv2.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)

    return image


def draw_pose_landmarks(
        image,
        landmarks,
        # upper_body_only,
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv2.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv2.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
            0] > visibility_th:
            cv2.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
            0] > visibility_th:
            cv2.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
            0] > visibility_th:
            cv2.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
            0] > visibility_th:
            cv2.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
            0] > visibility_th:
            cv2.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
            0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), 2)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][
            0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
            cv2.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
            0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
            cv2.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
            0] > visibility_th:
            cv2.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
            0] > visibility_th:
            cv2.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
            0] > visibility_th:
            cv2.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
            cv2.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
            0] > visibility_th:
            cv2.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 255, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
            0] > visibility_th:
            cv2.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
            0] > visibility_th:
            cv2.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
            cv2.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
            0] > visibility_th:
            cv2.line(image, landmark_point[11][1], landmark_point[23][1],
                    (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
            cv2.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
            cv2.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                0] > visibility_th:
                cv2.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                0] > visibility_th:
                cv2.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                0] > visibility_th:
                cv2.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                0] > visibility_th:
                cv2.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                0] > visibility_th:
                cv2.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                0] > visibility_th:
                cv2.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                0] > visibility_th:
                cv2.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                0] > visibility_th:
                cv2.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image