### 実行方法
___

`$ pip install mediapipe opencv-python scipy`

`$ python main.py`

config.pyにて、実行モードや設定変更可能


### 以下、所感など
___

### MediaPipeについて

#### 顔認識・ボディトラッキング
![](images/demo_body_track_Moment.jpg)

顔の追跡や、手足の追跡、位置情報の推定までをtensorflow-liteを使って行えるOSSです。
モデルもとても軽く、ラズパイでもしっかり追従してくれるのが特徴。

今回もPCはGPU処理なしで動かしていましたが、全然余裕で動きました。


#### 虹彩検出
![](images/demo_eye_track_Moment.jpg)

虹彩検出については、他のライブラリや記事から拾ったものも試してみたが、
こちらの精度が非常によかった。
MediaPipe >= 0.8.8で実行可能。


### 新規開発要素
___

#### 虹彩位置推定


#### まぶたの開度推定


#### 猫耳＆猫ひげオーバーレイ

やっぱり盛るカメラといえば、SNOWみたいな猫耳フィルタだよね。
ということで、機能を付けときました。

顔認識でトラッキングしている、おでこの点２点を原点に使って猫耳画像を幾何変換し、
猫耳を生やしています。

猫ひげも、鼻と頬っぺたのトラッキングされている点を原点として幾何変換することでオーバーレイしています。



#### キラキラエフェクトオーバーレイ

えふすとに無料公開されているエフェクトを読み込んで、
オーバーレイしているのみです。


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
