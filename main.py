# The algorithm that recover occluded picture
import time
import math
import os
from PIL import Image
import numpy as np
import re
import OMP
from matplotlib import pyplot as plt
# from sklearn.linear_model import OrthogonalMatchingPursuit

# 设定参数
# 下采样压缩后的列数
resizeCol = 40
# 下采样压缩后的行数
resizeRow = 55

# 从文件中根据文件名正则匹配得到训练数据和测试数据
def DivTrainTest(files):
    trains = []
    tests = []
    expr = re.compile("\S-\d*-(\d+?)\.pgm")
    for i in files:
        num = int(expr.findall(i)[0])
        if num < 14:
            if num <= 7:
                trains.append(i)
            else:
                tests.append(i)
        else:
            if num <= 20:
                trains.append(i)
            else:
                tests.append(i)
    return trains, tests

# 读取图像文件
def read_img(FilePath, downSample = True):
    # 打开文件
    im = Image.open(FilePath)
    # 判断是否下采样压缩
    if downSample:
        im = im.resize((resizeCol,resizeRow))

    # 默认为uint8,归一化时会溢出,所以需要更改类型
    im = np.array(im).astype(int).reshape(-1, 1)

    # # 归一化系数
    # division = math.sqrt(sum(im*im))
    # # 归一化
    # im = im / division
    # # 返回归一化结果和归一化系数
    # return im, division

    # 不归一化
    return im

# 读入训练数据
def readTrainData(DirPath, trainNum, trainTotal):
    A = []
    i = 0
    for root, dirs, files in os.walk(DirPath):
        TrainFiles, TestFiles = DivTrainTest(files)
        for f in TrainFiles:
            if i < trainNum:
                # 归一化
                # pic, div = read_img(os.path.join(root,f))

                # 不归一
                pic = read_img(os.path.join(root,f))

                A.append(pic)
            i = i+1
            if i >= trainTotal:
                i = 0
        return np.concatenate(A, axis=1)

# 获取测试数据
def getTestData(DirPath, testNum, testTotal):
    A = []
    i = 0
    for root, dirs, files in os.walk(DirPath):
        TrainFiles, TestFiles = DivTrainTest(files)
        for f in TestFiles:
            if i < testNum:
                A.append(os.path.join(root,f))
            i = i+1
            if i >= testTotal:
                i = 0
        return A

# 通过读入的训练数据构建字典
def getDictionary(TrainData):
    return np.concatenate([TrainData, np.identity(TrainData.shape[0])], axis=1)


# 构建字典
s = time.time()
TrainData = readTrainData(os.path.join(os.getcwd(),"test"), 7, 14)
A = getDictionary(TrainData)
e = time.time()
print("字典大小为",A.shape)
print("构建字典耗时",e-s, " s")

# TestData = getTestData(os.path.join(os.getcwd(),"test"), 4, 12)

# 测试
for i in TestData:
    # y, div = read_img(i)
    # 不归一化
    y= read_img(i)
    # Self-Written OMP
    s = time.time()
    result = OMP.OMP(A, y, A.shape[0])
    e = time.time()
    print("解决该问题耗时",e-s, " s")
    # 误差系数
    e = result[-A.shape[0]:].reshape(-1,1)
    # 表达系数
    coe = result[0:-A.shape[0]].reshape(-1,1)
    print("表达系数:\n",coe)
    print("误差系数:\n",e)
    # # 归一化
    # # 原图
    # rawImg = Image.fromarray(np.uint8((y*div).reshape((55,40)))).convert("RGB")
    # # 去遮挡恢复的图片
    # recoverImg = Image.fromarray(np.uint8(((y - e)*div).reshape((55,40)))).convert("RGB")
    # 不归一化
    # 原图
    rawImg = Image.fromarray(np.uint8((y).reshape((55,40)))).convert("RGB")
    # 去遮挡恢复的图片
    recoverImg = Image.fromarray(np.uint8((y - e).reshape((55,40)))).convert("RGB")

    plt.subplot(1,2,1)
    plt.imshow(rawImg)
    plt.subplot(1,2,2)
    plt.imshow(recoverImg)
    plt.show()
    # Image.fromarray(np.uint8(y - e).reshape((resizeRow,resizeCol))).show()

# y, div = read_img("d:\GitRepository\SparseRepresentation\\test\m-001-13.pgm")
# y = read_img("d:\GitRepository\SparseRepresentation\\test\m-001-13.pgm")

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


# # Self-Written OMP
# s = time.time()
# result = OMP.OMP(A, y, A.shape[0])
# e = time.time()
# print("Solving the problem cost",e-s, " s")
# e = result[-A.shape[0]:].reshape(-1,1)
# coe = result[0:-A.shape[0]].reshape(-1,1)
# print(coe)
# print(e)
# # Image.fromarray(np.uint8(((y - e)*div).reshape((55,40)))).show()
# Image.fromarray(np.uint8(y - e).reshape((resizeRow,resizeCol))).show()
