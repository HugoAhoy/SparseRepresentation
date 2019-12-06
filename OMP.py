import numpy as np
import copy

def LSFit(A, y):
    return np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, y))

def findPos(vec):
    idx = 0
    for i in range(len(vec)):
        if(abs(vec[i]) > abs(vec[idx])):
            idx = i
    return idx

def OMP(A, y, iter_num):
    k = 0
    # 深拷贝
    res = copy.deepcopy(y)
    Allbase = copy.deepcopy(A)
    
    # 基的集合
    base = np.zeros([A.shape[0],0])
    pos = []
    while k < iter_num:
        # 残差在各个基上的投影
        proj = np.matmul(Allbase.T, res)

        # 获得投影长度最大的基的索引并加入基的集合
        idx = findPos(proj)
        base = np.column_stack((base, Allbase[:, idx]))
        pos.append(idx)

        # 将原来对应的基置0,这样在选择基的时候就相当于删除了
        Allbase[:, idx] = 0
        k = k + 1

        # 最小二乘重新计算表达和残差
        x = LSFit(base, y)
        res = y - np.matmul(base, x)
    signal = np.zeros(A.shape[1])
    x = LSFit(base, y)
    for i in range(len(pos)):
        signal[pos[i]] = x[i]
    return signal

if __name__ == "__main__":
    A = np.array([[-0.707, 0.8, 0],[0.707, 0.6, -1]])
    y = np.array([[1.65],[-0.25]])
    x = OMP(A, y, A.shape[0])
    print(x)