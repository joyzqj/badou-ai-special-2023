# 处理顺序
# 1.指定信噪比SNR（信号和噪声所占比例），取值范围在[0,1]之间
# 2.计算总像素数目SP，得到要加噪的像素数目NP=SP*SNR
# 3.随机获取要加噪的每个像素位置P（i，j）
# 4.指定像素值为0或者255
# 5.重复3,4两个步骤完成所有NP个像素的加噪
import  numpy as np
import  random
import cv2
from  numpy import shape

def SPNoise(src,persnetage):
    noise_img=src
    noise_num=int(persnetage*src.shape[0]*src.shape[1])#计算噪声数量
    for i in range(noise_num):
        random_x=random.randint(0,src.shape[0]-1)#不处理图像边缘
        random_y=random.randint(0,src.shape[1]-1)
        if random.random() < 0.5:
            noise_img[random_x, random_y] =0
        else :
            noise_img[random_x, random_y] =255
    return noise_img


# 测试
img=cv2.imread("../data/lenna.png",0)
noise_img=SPNoise(img,0.2)
img=cv2.imread("../data/lenna.png")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("../temp/SP_noise.png",noise_img)
cv2.imshow("img",img_gray)
cv2.imshow("SPnoise",noise_img)
cv2.waitKey(0)

# # 调接口
# from skimage import util
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
#
# img1 = Image.open(r"../data/lenna.png")
# img = np.array(img1)
# noisy = util.random_noise(img, mode='s&p')
# plt.title('sp')
# plt.xticks([])   # remove ticks
# plt.yticks([])
# plt.imshow(noisy)
# plt.show()