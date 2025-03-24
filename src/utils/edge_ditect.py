import cv2
import numpy as np


# ガウシアン差分フィルタリング
def DoG(img, kernel_size, sigma, k=1.6, gamma=1):
    """
    ラプラシアンフィルタはノイズに弱いため、ガウシアンフィルタを用いて高周波成分をブラーにする
    DoGはガウシアンフィルタの差分を用いてラプラシンアンガウシアンフィルタを差分する
    近傍のピクセルからその画素を暗くすべきか，もしくは明るくすべきかという情報が与えられている

    :param kernel_size: カーネルサイズ （ボカシの粒度）
    """
    g1 = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    g2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma * k)
    return g1 - gamma * g2


def XDoG(image, kernel_size, sigma, epsilon, phi, k=1.6, gamma=0.98):
    """
    イプシロンよりもDog_imgが大きい場合は1にする
    :param epsilon: しきい値
    """
    epsilon /= 255
    Dog_img = DoG(image, kernel_size, sigma, k, gamma)
    Dog_img /= Dog_img.max()

    # tanh関数 = 活性化関数
    e = 1 + np.tanh(phi * (Dog_img - epsilon))
    e[e >= 1] = 1

    return e
