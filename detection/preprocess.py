import cv2


def preprocess_eye(image):

    # 顔を検出できない場合の真っ黒画像の場合エラーになるのを無理矢理回避
    try:
        # リサイズ
        rate = 10
        image = cv2.resize(image, dsize=None, fx=rate, fy=rate)

        # 白黒変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ２値化
        # threshold = 70
        # ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        per = 0.4
        image = p_tile_threshold(image, per)

    except:
        pass

    return image


def p_tile_threshold(img_gry, per):
    """
    Pタイル法による2値化処理
    :param img_gry: 2値化対象のグレースケール画像
    :param per: 2値化対象が画像で占める割合
    :return img_thr: 2値化した画像
    """

    # ヒストグラム取得
    img_hist = cv2.calcHist([img_gry], [0], None, [256], [0, 256])

    # 2値化対象が画像で占める割合から画素数を計算
    all_pic = img_gry.shape[0] * img_gry.shape[1]
    pic_per = all_pic * per

    # Pタイル法による2値化のしきい値計算
    p_tile_thr = 0
    pic_sum = 0

    # 現在の輝度と輝度の合計(高い値順に足す)の計算
    for hist in img_hist:
        pic_sum += hist

        # 輝度の合計が定めた割合を超えた場合処理終了
        if pic_sum > pic_per:
            break

        p_tile_thr += 1

    # Pタイル法によって取得したしきい値で2値化処理
    ret, img_thr = cv2.threshold(img_gry, p_tile_thr, 255, cv2.THRESH_BINARY)

    return img_thr
