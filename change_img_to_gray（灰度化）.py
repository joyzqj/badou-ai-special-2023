#彩色图像灰度化的步骤
import cv2
import  numpy as np
import  matplotlib.pyplot as plt
# 1.使用imread函数读取一张彩色图
img=cv2.imread("../data/lenna.png")
# 2.获取图像的宽高shape
height,width=img.shape[:2]
# 3.创建一张与原图大小一致的全为0的单通道图
img_gray=np.zeros([height,width],img.dtype)
# 4.遍历图像的宽高，取出当前图像宽高的bgr坐标
for i in range(height):
    for j in range(width):
        m=img[i,j]
# 4.将bgr坐标转换为gray坐标并赋值给新图像
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.imwrite("../temp/img_to_gray.png",img_gray)

# 调接口
# 1.读取一张彩色图
# img=cv2.imread("../data/lenna.png")
# # 2.灰度化
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray",img_gray)
# cv2.waitKey(0)