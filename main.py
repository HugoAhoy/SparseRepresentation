# The algorithm that recover occluded picture
import time
# from numba import autojit
# from sklearn.linear_model import orthogonal_mp
from functools import reduce
import os
import re
from PIL import Image
import numpy as np

def read_img(FilePath):
    im = Image.open(FilePath)    # 读取文件
    im = np.array(im).reshape(-1, 1)
    # print(im, im*im)
    im = im / (reduce(sum, np.nditer(im*im)))
    return im

# getDictionary A
def getDictionary(TrainData):
    return np.concatenate([TrainData, np.identity(TrainData.shape[0])], axis=1)

def readTrainData(DirPath, trainNum, testNum):
    A = []
    i = 0
    for root, dirs, files in os.walk(DirPath):
        for f in files:
            if i < trainNum:
                pic = read_img(os.path.join(root,f))
                A.append(pic)
            i = i+1
            if i >= trainNum+testNum:
                i = 0
        return np.concatenate(A, axis=1)

# s = time.time()
# TrainData = readTrainData(os.path.join(os.getcwd(),"AR"), 14, 12)
# A = getDictionary(TrainData)
# e = time.time()
# print(A.shape)
# print("build dic cost ",e-s, " s")
# y = read_img("D:\GitRepository\SparseRepresentation\AR\m-001-25.pgm")
x = np.array([1,2,3,4,5,6,7,8,9])
print(x)
print(x*x)
# result = orthogonal_mp(A, y, precompute=True)
# print(result)