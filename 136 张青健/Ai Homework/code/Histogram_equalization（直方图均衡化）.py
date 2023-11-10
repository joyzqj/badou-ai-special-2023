#直方图均衡化：
'''
原理：直方图均衡化是将原图像的直方图通过变换函数变为均匀的直方图，然后按均匀直方图修改原
图像，从而获得一幅灰度分布均匀的新图像。
作用：图像增强
'''
# 步骤：
import  cv2
import  numpy as np
import  matplotlib.pyplot as plt
# 1. 依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
# img= cv2.imread("../data/lenna.png")
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 2. 计算灰度直方图的累加直方图
# 3. 根据累加直方图和直方图均衡化原理得到输入与输出之间的映射关系。
# 4. 最后根据映射关系得到结果：dst(x,y) = H'(src(x,y))进行图像变换
# 灰度图像的直方图
# 方法一
# plt.figure()
# plt.hist(img_gray.ravel(),256)
# plt.show()

# # 方法二
# hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
# plt.figure()
# plt.plot(hist)
# plt.xlim([0,256])
# plt.show()

# 彩色图直方图
# img=cv2.imread("../data/lenna.png")
# channels=cv2.split(img)
# colors=("b","g","r")
# plt.figure()
# for (channel,color) in zip(channels,colors):
#     hist=cv2.calcHist([channel],[0],None,[256],[0,256])
#     plt.plot(hist,color=color)
#     plt.xlim([0,256])
# plt.show()

# 彩色图直方图均衡化前后对比
# img = cv2.imread("../data/lenna.png", 1)
# cv2.imshow("src", img)
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("../temp/dst_rgb", result)
# cv2.waitKey(0)

# # 灰度图像直方图均衡化前后对比
img = cv2.imread("../data/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
