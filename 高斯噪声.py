#高斯采样分布公式
# (输出像素)P_out=（输入像素）P_in + random.gauss(mu=mean,sigma=sigma)
# 处理顺序：
import  cv2
import  random
import numpy as np
import matplotlib.pyplot as plt
# 1.输入参数sigma，mean
# 2.生成高斯随机数
# 3.根据输入像素计算输出像素
# 4.重新将像素值缩放在[0,255]之间
# 5.循环所有像素
# 6.输出图像
def GaussNoise(src,means,sigma,percentage):
    noise_img=src#复制图片
    noise_num=int(percentage*src.shape[0]*src.shape[1])#计算噪声数量
    for i in range(noise_num):
        random_x=random.randint(0,src.shape[0]-1)#不处理图像边缘，故-1
        random_y=random.randint(0,src.shape[1]-1)
        #在原有灰度像素上加上随机数
        noise_img[random_x,random_y] +=random.gauss(means,sigma)#与椒盐噪声相区别的一行
        #若灰度值小于0则赋值为0，若灰度值大于255，则赋值为255
        if noise_img[random_x,random_y]<0:
            noise_img[random_x, random_y]=0
        elif noise_img[random_x,random_y]>255:
            noise_img[random_x, random_y] = 255
    return noise_img

# 测试
img=cv2.imread("../data/lenna.png",0)
noise_img=GaussNoise(img,2,20,0.8)
img=cv2.imread("../data/lenna.png")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("../temp/Gauss_noise.png",noise_img)
cv2.imshow("img",img_gray)
cv2.imshow("gauss",noise_img)
cv2.waitKey(0)


# # 调接口
# from skimage import util
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
#
# img1 = Image.open(r"../data/lenna.png")
# img = np.array(img1)
# noisy = util.random_noise(img, mode='gaussian', mean=0, var=0.2)
# plt.title('Gauss')
# plt.xticks([])   # remove ticks
# plt.yticks([])
# plt.imshow(noisy)
# plt.show()