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
        img_gray[i,j]=int(m[0]/255+m[1]/255)
        if img_gray[i,j]<=0.5:
            img_gray[i,j]=0
        else:
            img_gray[i, j] =255

print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.imwrite("../temp/img_to_binary.png",img_gray)
cv2.waitKey(0)

# # 调接口
# # 1.以灰度化方式读取一张彩色图
# img=cv2.imread("../data/lenna.png",cv2.IMREAD_GRAYSCALE)
# # 2.设定一个阈值和最大像素值
# binary_balue=128
# binary_max=255
# # 3.二值化
# ret,binary=cv2.threshold(img,binary_balue,binary_max,cv2.THRESH_BINARY)
# cv2.imshow("binary",binary)
# cv2.imwrite("../temp/img_to_binary1.png",binary)
# cv2.waitKey(0)