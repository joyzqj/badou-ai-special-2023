# # 1.图像平移
# import  cv2
# import numpy as np
# img=cv2.imread("../data/lenna.png",1)
# h,w,channels=img.shape
# # 将图像向右平移100个像素点，向下平移100个像素点
# change_img=np.float32([[1, 0, 100], [0, 1, 100]])
# # 进行2D 仿射变换
# translated = cv2.warpAffine(img, change_img, (w, h))
# cv2.imshow("img",img)
# cv2.imshow("translated_img",translated)
# cv2.imwrite('../temp/translated.jpg', translated)  #  变换后图像保存的路径
# cv2.waitKey(0)


# 2、图像旋转
# （1）概念：将图像绕图像中心逆时针或顺时针旋转一定的角度
# 注意：逆时针为正，顺时针为负
# cv2.getRotationMatrix2D(center, angle, scale)
    # center：旋转中心点 (cx, cy) ，此中心点可以随意指定为图像中的任意像素点
    # angle：旋转的角度，逆时针方向为正方向 ， 角度为正值代表为逆时针旋转
     # scale：缩放倍数，值等于1.0代表尺寸不变


# import numpy as np
# import cv2
# from math import cos,sin,radians
# from matplotlib import pyplot as plt
# img=cv2.imread("../data/lenna.png",1)
# h,w,channel=img.shape
# # 求旋转图像的中心点
# c_x=w//2
# c_y=h//2
# center=(c_x,c_y)
# new_dim=(w,h)
# # 进行2D 仿射变换
# # 围绕原点 逆时针旋转30度
# trsans_30 = cv2.getRotationMatrix2D(center=center,angle=30, scale=1.0)
# rotated_30 = cv2.warpAffine(img, trsans_30, new_dim)
#
# # 围绕原点 逆时针旋转45度
# trsans_45 = cv2.getRotationMatrix2D(center=center,angle=45, scale=1.0)
# rotated_45 = cv2.warpAffine(img, trsans_45, new_dim)
#
# # 围绕原点 逆时针旋转60度
# trsans_60 = cv2.getRotationMatrix2D(center=center,angle=60, scale=1.0)
# rotated_60 = cv2.warpAffine(img, trsans_60, new_dim)
# cv2.imshow("trsans_30",rotated_30)
# cv2.imwrite("../temp/trsans_30.png",rotated_30)
# cv2.imshow("trsans_45",rotated_45)
# cv2.imwrite("../temp/trsans_45.png",rotated_45)
# cv2.imshow("trsans_60",rotated_60)
# cv2.imwrite("../temp/trsans_60.png",rotated_60)
# cv2.waitKey(0)

# 3、图像缩放
# 1）利用x方向与y方向的两个缩放系数控制图像缩小或放大一定的比例
# opencv实现方法：resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
# src：输入图片
# dsize：输出图片的尺寸
# dst：输出图片
# fx：x轴的缩放因子
# fy：y轴的缩放因子
# interpolation：插值方式
# INTER_NEAREST - 最近邻插值
# INTER_LINEAR - 线性插值（默认）
# INTER_AREA - 区域插值
# INTER_CUBIC - 三次样条插值
# INTER_LANCZOS4 - Lanczos插值

# import cv2
# import numpy as np
# img=cv2.imread("../data/lenna.png",1)
# h,w,channel=img.shape
# resize_img=cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("yuantu",img)
# cv2.imshow("resize_to_800x800",resize_img)
# cv2.imwrite("../temp/resize_img.png",resize_img)
# cv2.waitKey(0)

# # 使用仿射矩阵实现
# import numpy as np
# import cv2
# img=cv2.imread("../data/lenna.png",1)
# h,w,channel=img.shape
# # x轴焦距 1.5倍
# fx = 1.5
# # y轴焦距 2倍
# fy = 1.5
# # 声明变换矩阵 x方向放大2倍，y方向放大2倍
# M = np.float32([[fx, 0, 0], [0, fy, 0]])
# # 进行2D 仿射变换
# resized = cv2.warpAffine(img, M, (int(w*fx), int(h*fy)))
# cv2.imshow("yuantu",img)
# cv2.imshow("resize_to_800x800",resized)
# cv2.imwrite("../temp/resize_img1.png",resized)
# cv2.waitKey(0)


# 4、图像错切
# 在某方向上，按照一定的比例对图形的每个点到某条平行于该方向的直线的有向距离做放缩得到的平面图形，
# 通常分为x方向与y方向的错切
# import cv2
# import numpy as np
# def shear(img, x=False, y=False, shear=10):
#     """
#     仿射变换：sheer
#     :param image:
#     :param x:           按 x 方向sheer
#     :param y:           按 y 方向sheer
#     :param shear:       sheer 角度
#     :return:
#     """
#     # Shear
#     # 设置 裁剪 的仿射矩阵系数
#     S = np.eye(3)
#     tan = np.tan(shear * np.pi / 180)
#     if x is True:
#         S[0, 1] = tan  # x shear (deg)
#     if y is True:
#         S[1, 0] = tan  # y shear (deg)
#     M = S[0:2]
#     height, width,channel = img.shape
#     scale = 1.0
#     image = cv2.warpAffine(img, M, (int(width * scale), int(height * scale)))
#     return image
#
# img=cv2.imread("../data/lenna.png",1)
# shear_img = shear(img, True)
# cv2.imshow("yuantu",img)
# cv2.imshow("cuofen_img",shear_img)
# cv2.imwrite("../temp/cuofen_img.png",shear_img)
# cv2.waitKey(0)


# 5、图像翻转
# （1）概念：将图像沿水平/垂直或水平垂直方向做镜面的翻转，即图像沿水平/垂直方向翻转前后是对称的
# flip(src, flipCode[, dst]) -> dst
# src：输入图片
# flipCode：翻转代码
# 1 水平翻转 Horizontally （图片第二维度是column）
# 0 垂直翻转 *Vertically * （图片第一维是row）
# -1 同时水平翻转与垂直反转 Horizontally & Vertically

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# img=cv2.imread("../data/lenna.png",1)
#
# def bgr2rbg(img):
#     '''
#         将颜色空间从BGR转换为RBG
#     '''
#     return img[:,:,::-1]
#
# # 水平翻转
# flip_h = cv2.flip(img, 1)
# # 垂直翻转
# flip_v = cv2.flip(img, 0)
# # 同时水平翻转与垂直翻转
# flip_hv = cv2.flip(img, -1)
# cv2.imshow("flip_h",flip_h)
# cv2.imshow("flip_v",flip_v)
# cv2.imshow("flip_hv",flip_hv)
# cv2.imwrite('../temp/flip_h.jpg', flip_h)
# cv2.imwrite('../temp/flip_h.jpg', flip_v)
# cv2.imwrite('../temp/flip_h.jpg', flip_hv)
# cv2.waitKey(0)


# 6、图像裁剪
# （1）概念：裁剪出图像中的某一部分子区域
# import cv2

'''
    x_start: 图像height的起始像素位置
    x_range: 图像裁剪后的height
    y_start: 图像width的起始像素位置
    y_range: 图像裁剪后的width
'''
# def crop(img, x_start, x_range, y_start, y_range):
#     #裁剪
#     crop_img = img[x_start:x_start+x_range, y_start:y_start+y_range]
#     return crop_img
#
# '''
#     从图像中心裁剪出一个区域
# '''
# def crop_center(img, new_height, new_width):
#     #裁剪
#     height, width, channel = img.shape
#     center_x = int(height / 2)
#     center_y = int(width / 2)
#     crop_img = img[center_x - int(new_height/2):center_x + int(new_height/2), center_y - int(new_width/2):center_y + int(new_width/2)]
#     return crop_img

# img=cv2.imread("../data/lenna.png",1)
# print(img.shape)
# # crop_img = crop(img, 500, 1000, 1000, 2000)
# crop_img = crop_center(img, 200, 200)
# print(crop_img.shape)
# cv2.imshow("crop_center",crop_img)
# cv2.imwrite("../temp/crop_center.jpg", crop_img)
# cv2.waitKey(0)

#
# 7、图像填充
# （1）概念：对图像四周用颜色进行填充
# 上下各填充100像素，左右各填充50像素，填充值为(0, 0, 0)，生成新的图像
# import cv2
#
# def pad(img, top, bottom, left, right, value):
#     pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
#     return pad_img
# img=cv2.imread("../data/lenna.png",1)
# pad_img = pad(img, 100, 100, 50, 50, (0, 0, 0))
# cv2.imshow("yuantu",img)
# cv2.imshow("pad",pad_img)
# cv2.imwrite("../temp/pad.jpg", pad_img)
# cv2.waitKey(0)

# 8、补充
# （1）概念：cv2.getAffineTransForm()通过找原图像中三个点的坐标和变换图像的相应三个点坐标，创建一个2X3的变换矩阵M，作为函数
# cv2.warpAffine(image, M, (image.shape[1], image.shape[0])中的参数。
# cv2.getAffineTransform(pts1,pts2)
# pts1：原图像三个点的坐标
# pts2：原图像三个点在变换后相应的坐标

import cv2
import numpy as np

img=cv2.imread("../data/lenna.png",1)
rows, cols, channel = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow("yuantu",img)
cv2.imshow("buchon",dst)
cv2.imwrite("../temp/buchon.jpg", dst)
cv2.waitKey(0)
