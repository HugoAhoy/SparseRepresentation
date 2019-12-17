import numpy as np
import os

# 最小二乘
def LSFit(A, y):
    return np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, y))

# 用伪逆最小二范数
def LSwithPinv(A, y):
    return np.matmul(np.linalg.pinv(np.matmul(A.T, A)), np.matmul(A.T, y))

# OMP算法，A为字典，y为待表达信号，iter_num为迭代次数
def OMP(A, y, iter_num):
    k = 0
    # 初始化残差
    res = np.array(y, dtype=np.float64)
    # 初始化所有的基
    Allbase = np.array(A, dtype=np.float64)

    # 选择的基的集合
    base = np.zeros([A.shape[0],0], dtype=np.float64)

    # 用于记录选择的基在所有基中的位置
    pos = []

    # 迭代
    while k < iter_num:
        # 残差在各个基上的投影
        proj = np.matmul(res.T,Allbase)

        # 获得投影长度最大的基的索引并加入基的集合
        # idx = findPos(proj)
        idx = np.argmax(abs(proj))
        base = np.column_stack((base, Allbase[:, idx]))
        pos.append(idx)

        # 将原来对应的基置0,这样在选择基的时候就相当于删除了,不会被重复选择
        Allbase[:, idx] = 0
        k = k + 1

        # 最小二乘重新计算表达
        # x = LSFit(base, y)

        # 伪逆计算最小二范数重新计算表达
        x = LSwithPinv(base, y)

        # 计算残差
        res = y - np.matmul(base, x)
        # print("x\n",x)
        # print("base\n",base)
        # print("res\n", res)
        # os.system("pause")
        print(k)
    
    # 信号构成
    signal = np.zeros(A.shape[1], dtype=np.float64)
    
    # 计算最终表达
    x = LSwithPinv(base, y)

    # 生成最终稀疏信号
    for i in range(len(pos)):
        signal[pos[i]] = x[i]
    return signal

# 单元测试
if __name__ == "__main__":
    A = np.array([[-0.707, 0.8, 0],[0.707, 0.6, -1]])
    y = np.array([[1.65],[-0.25]])
    x = OMP(A, y, A.shape[0])
    print(x)