# The algorithm that recover occluded picture
import time
import math
import os
from PIL import Image
import numpy as np
import re
import OMP
# from sklearn.linear_model import OrthogonalMatchingPursuit

# 设定参数
resizeCol = 40
resizeRow = 55

# 从文件中根据文件名正则匹配得到训练数据
def getTrain(files):
    result = []
    expr = re.compile("\S-\d*-(\d+?)\.pgm")
    for i in files:
        num = int(expr.findall(i)[0])
        if num < 14:
            if num <= 7:
                result.append(i)
        else:
            if num <= 20:
                result.append(i)
    return result

# 读取图像文件
def read_img(FilePath, downSample = True):
    im = Image.open(FilePath)    # 打开文件
    if downSample:
        im = im.resize((resizeCol,resizeRow))

    # 默认为uint8,归一化时会溢出,所以需要更改类型
    im = np.array(im).astype(int).reshape(-1, 1)
    # division = math.sqrt(sum(im*im))
    # im = im / division
    # 返回归一化结果和归一化系数
    # return im, division
    return im

# 读入训练数据
def readTrainData(DirPath, trainNum, trainTotal):
    A = []
    i = 0
    for root, dirs, files in os.walk(DirPath):
        files = getTrain(files)
        for f in files:
            if i < trainNum:
                # pic, div = read_img(os.path.join(root,f))
                pic = read_img(os.path.join(root,f))

                A.append(pic)
            i = i+1
            if i >= trainTotal:
                i = 0
        return np.concatenate(A, axis=1)


# 通过读入的训练数据构建字典
def getDictionary(TrainData):
    return np.concatenate([TrainData, np.identity(TrainData.shape[0])], axis=1)


# 构建字典
s = time.time()
TrainData = readTrainData(os.path.join(os.getcwd(),"test"), 10, 14)
A = getDictionary(TrainData)
e = time.time()
print(A.shape)
print("build dic cost",e-s, " s")

# y, div = read_img("d:\GitRepository\SparseRepresentation\\test\m-001-13.pgm")
y = read_img("d:\GitRepository\SparseRepresentation\\test\m-001-13.pgm")

# model = OrthogonalMatchingPursuit(n_nonzero_coefs=A.shape[1])

# s = time.time()
# result = model.fit(A, y).coef_
# e = time.time()
# print("Solving the problem cost",e-s, " s")
# print(result)
# print(result.shape)
# e = result[-A.shape[0]:].reshape(-1,1)
# print(e)
# coe = result[0:-A.shape[0]].reshape(-1,1)
# print(coe)
# # Image.fromarray(np.uint8((div*(y - e)).reshape((resizeRow,resizeCol)))).show()
# Image.fromarray(np.uint8((y - e).reshape((resizeRow,resizeCol)))).show()


# Self-Written OMP
s = time.time()
result = OMP.OMP(A, y, A.shape[0])
e = time.time()
print("Solving the problem cost",e-s, " s")
e = result[-A.shape[0]:].reshape(-1,1)
coe = result[0:-A.shape[0]].reshape(-1,1)
print(coe)
print(e)
# Image.fromarray(np.uint8(((y - e)*div).reshape((55,40)))).show()
Image.fromarray(np.uint8(y - e).reshape((resizeRow,resizeCol))).show()
