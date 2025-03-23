import cv2
# import numpy as np


# ガウシアン差分フィルタリング
def DoG(img, size, sigma, k=1.6, gamma=1):
    """
    ラプラシアンフィルタはノイズに弱いため、ガウシアンフィルタを用いて高周波成分をブラーにする
    DoGはガウシアンフィルタの差分を用いてラプラシンアンガウシアンフィルタを差分する

    :param size: カーネルサイズ （ボカシの粒度）
    """
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma * k)
    return g1 - gamma * g2
