# Image_Comparison
2枚の画像の差分検出パッケージ（開発途中)  
Canny法[Canny 1986]による特徴点検出の手法を用いて,2枚の画像を比較し差分検出を行うパッケージです．  

## はじめに  
**本パッケージは開発途中のものですので，2つの画像の差分検出の精度に関しては保証しかねます．**

## 開発環境
以下の環境で開発，動作確認を行なっています．
* OS
  - macOS Catalina 10.15.2
* 使用言語
  - python3
* 使用パッケージ
  - NumPy
  - OpenCV
  - Matplotlib  

画像の形式はPNG形式です．

## インストール
以下のコマンドを作業ディレクトリ内で実行し,本リポジトリをダウンロードしてください．  
python3と使用パッケージのインストールについては省略します．
`$ git clone https://github.com/takahara3/image_comparison.git`

## 実行方法(canny.pyの場合)
1. imgディレクトリ内に差分検出を行いたい2つの画像を入れます．
    - 比較元の画像名:  
    `origin_image.png`
    - 比較対象の画像名:  
    `comparison_image.png`
2. 以下のコマンドを実行します　　
    - 検出結果を表示：  
    `$ python3 canny.py`
    - 検出結果を表示，保存：  
    `$ python3 canny.py -s True`　　

imgディレクトリ内にサンプル画像が入っているので，サンプルでの実行結果を以下に示します．  
### サンプルでの実行結果  
* 比較元画像  
![origin_image](https://user-images.githubusercontent.com/49555813/70996972-527b5400-2117-11ea-8045-c8f247f46e3b.png)  
* 比較対象画像  
![comparison_image](https://user-images.githubusercontent.com/49555813/70997028-6f178c00-2117-11ea-9af9-d3e0b8af1b7a.png)  
* 差分検出結果  
![result](https://user-images.githubusercontent.com/49555813/70995552-2b6f5300-2114-11ea-8310-701fda4ac1b4.png)
