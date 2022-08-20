### 実行方法
___

`$ pip install mediapipe opencv-python scipy`

`$ python main.py`

config.pyにて、実行モードや設定変更可能


### 以下、所感など
___

### MediaPipeについて

#### 顔認識・ボディトラッキング
![demo_body_track_Moment](https://user-images.githubusercontent.com/97094663/185735078-7f44d68f-dff2-4ac8-92ca-80bb35065342.jpg)

顔の追跡や、手足の追跡、位置情報の推定までをtensorflow-liteを使って行えるOSSです。
モデルもとても軽く、ラズパイでもしっかり追従してくれるのが特徴。


#### 虹彩検出
![Uploading demo_eye_track_Moment.jpg…]()

MediaPipe >= 0.8.8で実行可能。


### 参考
___

mediapipeでポーズ推定

http://cedro3.com/ai/mediapipe/

mediapipeで虹彩検出

https://github.com/Kazuhito00/mediapipe-python-sample

アルファブレンド

https://qiita.com/smatsumt/items/923aefb052f217f2f3c5



### 素材
___

- キラキラエフェクト(えふすと)

- 猫素材（いらすとや）
