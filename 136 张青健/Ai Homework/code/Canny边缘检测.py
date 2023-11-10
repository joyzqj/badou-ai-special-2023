# 步骤
# 1.对图像进行灰度化
# 2.对图像进行高斯滤波
#     a.根据滤波的像素点及其领域点的灰度值按照一定的参数规则进行加权平均，这样可以有效滤去图像中叠加的高频噪声
# 3.检测图像的水平，垂直和对角边缘（如prewitt，sobel算子等）
# 4.对梯度幅值进行非极大值抑制，即寻找局部最大值，将非极大值所对应的灰度置为0，这样可以剔除一部分非边缘的点
# 5.用双阈值算法检测和连接边缘
import cv2
import numpy as np
img = cv2.imread("../data/lenna.png", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(img_gray, 50, 200)
cv2.imshow("canny",canny)
cv2.imwrite("../temp/canny.png",canny)
cv2.waitKey()
cv2.destroyAllWindows()

