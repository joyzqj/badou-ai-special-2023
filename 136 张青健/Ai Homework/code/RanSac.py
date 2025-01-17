# 一般RANSAC算法由两步骤迭代计算：
# （1）一个样本子集，包含数据选取（随机选取）。通过使用这些数据得到一个拟合模型和相关的模型参数。
# 样本子集的数量是最小充分的得到模型参数。
# （2）算法检查数据集中的哪些元素是一直在第一步估计到的模型当中的。
# 如果在阈值（相对噪声的最大偏离度）外的话，该模型元素不能拟合估计到的模型便会被当做outlier。
#
# inliers的设置称作“一致性设置”RANSAC算法会一直迭代直到获得足够的inliers。
#
# RANSAC的输入是一些观测数据和一些“可信度”参数，实现步骤：
#
# （1）随机选择一些原始数据，叫作假设inliers子集
# （2）建立模型拟合
# （3）用其他数据来验证，根据模型特定的loss-function来计算是否符合该模型
# （4）如果足够的点都算是“一致性”设置里则该模型算是好模型
# （5）比较所有的“一致性”设置（就是建立的所有模型）看看哪个inliers多就是我们要的。

import numpy as np
import scipy.linalg as sl
import scipy as sp

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）

    iterations = 0
    bestfit = nil #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k
    {
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)
        {
            if 满足maybemodel即error < t
                将点加入alsoinliers
        }
        if (alsoinliers样本点数目 > d)
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """
    iterations = 0
    best_fit = None
    best_err = np.inf  # 设置默认值
    best_inlier_indexs = None
    while iterations < k:  # 当迭代次数小于最大迭代数时
        maybe_indexs, test_indexs = random_partition(n, data.shape[0])
        print("test_indexs:", test_indexs)
        maybe_inliers = data[maybe_indexs, :]  # 获取size(maybe_indexs)行数据(x_i,y_i)
        test_points = data[test_indexs]  # 若干行(x_i,y_i)数据点
        maybe_model = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybe_model)  # 计算误差：平方和最小
        print("test_err:", test_err < t)

        also_indexs = test_indexs[test_err < t]
        print("also_index:", also_indexs)
        also_inliers = data[also_indexs, :]

        if debug:
            print('test_err.min:', test_err.min())
            print('test_err.max:', test_err.max())
            print('np.mean(test_err):', np.mean(test_err))
            print('iterations %d:len(also_inliers)=%d' % (iterations, len(also_inliers)))
        # if len(also_inliers)>d: #d 拟合较好时,需要的样本点最少的个数,当做阈值看待
        print('d=', d)
        if len(also_inliers) > d:
            better_data = np.concatenate((maybe_inliers, also_inliers), axis=None, out=None, dtype=None,
                                          )  # 样本连接array：(maybe_inliers,also_inliers)
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            new_err = np.mean(better_errs)  # 平均误差作为新的误差

            if new_err < best_err:
                best_fit = better_model
                best_err = new_err
                best_inlier_indexs = np.concatenate((maybe_indexs, also_indexs), axis=None, out=None, dtype=None,
                                                     )
            iterations += 1

    if best_fit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return best_fit, {'inliers': best_inlier_indexs}
    else:
        return best_fit


def random_partition(n, n_data):
    '''返回n行随机数以及n行随机数的长度'''
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class Liner_least_SquareModel:
    '''用最小二乘法求线性解，用于ransac的输入模型'''

    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resides, rank, s = sl.lstsq(A, B)  # 残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A, model)  # 计算的y值，B_fit=model.K*A+ model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 平方误差和
        return err_per_point

def test():
    #生成理想数据
    n_samples=500 #样本个数
    n_inputs=1 #输入变量个数
    n_outputs=1#输出变量个数
    A_exact=20*np.random.random((n_samples,n_inputs))#随机生成0-20之间的500个数据:行向量
    perfect_fit=60*np.random.normal(size=(n_inputs,n_outputs))#随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    #加入高斯噪声，最小二乘法能很好的处理
    A_noisy = A_exact + np.random.normal( size = A_exact.shape ) #500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal( size = B_exact.shape ) #500 * 1行向量,代表Yi

    if 1:
        #添加局外点
        n_outliers=100
        all_indexs=np.arange(A_noisy.shape[0])#获取索引0-499
        np.random.shuffle(all_indexs)#随机打乱all_indexs
        outlier_indexs=all_indexs[:n_outliers]#100个0-500的随机局外点
        A_noisy[outlier_indexs]=20*np.random.random((n_outliers,n_inputs))#加入噪声和局外点Xi
        B_noisy[outlier_indexs]=50*np.random.normal(size=(n_outliers,n_outputs)) #加入噪声和局外点的Yi

    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = Liner_least_SquareModel(input_columns,output_columns,debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = np.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    # ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 5e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_indexs, 0], B_noisy[outlier_indexs, 0], 'r.', label='outlier data')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ == "__main__":
        test()