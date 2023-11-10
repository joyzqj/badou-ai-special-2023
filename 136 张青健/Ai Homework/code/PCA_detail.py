# 特征选择步骤
# 1.生成过程:生成候选的特征子集
# 2.评价函数:评价特征子集的好坏
# 3.停止条件:决定什么时候该停止
# 4.验证过程：特征子集是否有效

#pca 步骤
# pca算法的优化目标：
# a.降维后同一维度的方差最大（求方差）
# b.不同维度之间的相关性（协方差即同一样本不同维度的协方差）为0,
# c.协方差矩阵的对角线就是各个维度上的方差
# d.样本矩阵的每行是一个样本，每列为一个维度，所以我们要按列计算均值

# 1.对原始数据0均值化（中心化）
# 2.求协方差矩阵
# 3.对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间
import  numpy as np
class CPCA(object):
    def __init__(self,x,k):
        self.x=x  #样本矩阵
        self.k=k  #k阶降维矩阵的k值
        self.center_x=[]  #矩阵x的中心化
        self.c=[]  #样本集的协方差矩阵
        self.w=[]  #样本矩阵x的降维转换矩阵
        self.z= [] #样本矩阵x的降维矩阵z

        self.center_x=self._centralized()
        self.c=self._cov()
        self.w=self._w()
        self.z=self._z()

    def _centralized(self):
        '''矩阵x的中心化'''
        print("样本矩阵x：\n",self.x)
        center_x=[]
        mean=np.array([np.mean(attr) for attr in self.x.T]) #样本集的特征均值
        print("样本集特征均值mean：\n",mean)
        center_x=self.x-mean #样本集中心化
        print("样本集中心化:\n",center_x)
        return center_x

    def _cov(self):
        '''求样本集x的协方差矩阵c'''
        ns=np.shape(self.center_x)[0]#样本集用例总数
        c=np.dot(self.center_x.T,self.center_x)/(ns-1) #求协方差矩阵
        print("协方差矩阵c：\n",c)
        return c

    def _w(self):
       a,b=np.linalg.eig(self.c)#特征值赋值给a，对应特征向量赋值给b
       print("协方差矩阵的特征值a：\n",a)
       print("协方差矩阵的特征向量b：\n",b)
       ind=np.argsort(-1*a) #给出特征值降序的topk的索引序列
       w_t = [b[:, ind[i]] for i in range(self.k)]
       w = np.transpose(w_t)
       print('%d阶降维转换矩阵w:\n' % self.k, w)
       return w

    def _z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        z = np.dot(self.x, self.w)
        print('x shape:', np.shape(self.x))
        print('w shape:', np.shape(self.w))
        print('z shape:', np.shape(z))
        print('样本矩阵X的降维矩阵z:\n', z)
        return z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    x = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    k = np.shape(x)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', x)
    pca = CPCA(x, k)






