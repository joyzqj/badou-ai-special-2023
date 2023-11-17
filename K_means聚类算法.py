# K_means聚类算法实现步骤
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 第一步：确定k值，即将数据集聚集成k个类簇或大小
# 第二步：从数据集中随机选择k个数据点做为质心或数据中心
# 第三步：分别计算每个点到质心之间的距离，并将每个点划分到离质心最近的距离
# 第四步：当每个质心都聚集了一些点后，重新定义算法选出新的质心，（对于每个簇，计算其平均值，即得到新的k个质心点）
# 第五步，迭代执行第三步到第四步，直到迭代终止条件满足为止（聚类结果不在变化

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

# 利用OpenCV实现k_means算法
# 读取原始图像的灰度颜色
img=cv2.imread("../data/lenna.png",0)
# 获取图像宽高
h,w=img.shape[:]
# 将二维像素转换为一维像素
data=img.reshape((h*w,1))
data=np.float32(data)
# 停止条件(type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 3.0)
# 设置标签
flags=cv2.KMEANS_RANDOM_CENTERS
#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, flags)
#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
   plt.savefig("../temp/k_means.png")
plt.show()
