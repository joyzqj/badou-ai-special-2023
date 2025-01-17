# 差值哈希算法
# 步骤
import  cv2
import numpy as np
def LinerHash(img):
    # 1. 缩放：图片缩放为8*9，保留结构，除去细节。
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    # 2. 灰度化：转换为灰度图。
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 3. 求平均值：计算灰度图所有像素的平均值。 ---这步没有，只是为了与均值哈希做对比
    # 4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
    hash_str=''
    for i in range(8):
        for j in range(8):
            if img_gray[i,j]>img_gray[i,j+1]:
                hash_str+='1'
            else:
                hash_str += '0'
    return hash_str
    # 八个差值，有8行，总共64位
    # 5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    # 6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样
    # 的，不相同位数越少，图片越相似。
#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

img1=cv2.imread('../data/lenna.png')
img2=cv2.imread('../data/lenna_noise.png')
hash1= LinerHash(img1)
hash2= LinerHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：',n)