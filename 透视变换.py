# 透视变换对畸变图像的校正需要取得畸变图像的一组4个点的坐标，
# 和目标图像的一组4个点的坐标，通过两组坐标点可以计算出透视变换的变换矩阵，
# 之后对整个原始图像执行变换矩阵的变换，就可以实现图像校正。
import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

if __name__=="__main__":
    img = cv.imread('../data/lenna.png',1)
    #修改颜色通道
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    rows,cols,ch = img.shape
    print(rows,cols,ch)

    # 定义我们要透视变换的点，即上面图的四个点 分别是原图的 左上 右上 左下 右下， 注意需要将四个点的坐标转换成float32
    pts1 = np.float32([[0,0],[0,500],[500,0],[500,500]])
    # 定义我们将图片展平的点，本次展平为一张图片
    pts2 = np.float32([[30,50],[30,400],[500,70],[400,500]])
    # 计算得到转化矩阵  输入的参数分别是原图像的四边形坐标 变换后图片的四边形坐标
    M = cv.getPerspectiveTransform(pts1,pts2)
    # 得到透视变换的图片
    dst = cv.warpPerspective(img,M,(400,400))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.savefig("../temp/fangshebianhuan_img.png")
    plt.show()
