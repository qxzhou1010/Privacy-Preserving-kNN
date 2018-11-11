#coding=utf-8
#Author:Qixian Zhou
"""
循环，单个点到所有点欧式距离的计算
然后分类预测
"""
import numpy as np
import mvhe
from time import clock
import time

def accRate(testDataLables, testDataLabelsPredicts ):
    accNum = 0
    for i in range( testDataLables.shape[0]):
        # print(testDataLabels - testDataLabelsPredicts)
        if (testDataLables - testDataLabelsPredicts)[i][0] == 0:
            accNum += 1
    print( "the predict correct rate is %.2f %%" %((accNum / testDataLables.shape[0] )* 100) )


def getK(sort_dist, k):
    k_dist = np.zeros((k,2))
    # sort_dist = all_Dist(train_data, test_data[0])
    for i in range(k):
        k_dist[i] = sort_dist[i]
    # print(k_dist)
    return k_dist

#---------------数据读取和处理--------------
train_data = np.loadtxt("D_train.txt")
trainDataNums, dim = train_data.shape
trainData = np.zeros( (trainDataNums, dim), dtype=object)   #主要想把所有的数据类型都转换为[class:int]
for i in range(trainDataNums):
    for j in range(dim):
        trainData[i][j] = int(train_data[i][j] * (10 ** 4)) #保证输入为class:int



test_data = np.loadtxt("D_test.txt")
testDataNums, dim = test_data.shape
testData = np.zeros( (testDataNums, dim), dtype=object)
for i in range(testDataNums):
    for j in range(dim):
        testData[i][j] = int(test_data[i][j] * (10 ** 4))   #保证输入为class：int
trainData_Tag = np.loadtxt( "D_train_result.txt")
testData_Tag = np.loadtxt( "D_test_result.txt")
testData_Tag.resize( (testDataNums,1))






def DataUpload(trainData):
    trainDataNums,dim = trainData.shape
    #数据构造
    trainDataReshape = np.zeros((trainDataNums, dim + 2), dtype=object)
    for i in range(trainDataNums):
        trainDataReshape[i][0] = 1
        temp = trainData[i].dot(trainData[i].T)
        trainDataReshape[i][1] = temp
        for j in range(dim):
            trainDataReshape[i][j + 2] = trainData[i][j]
    dim = trainDataReshape.shape[1] #维度更新
    #加密
    #   初始化参数
    T_cols = 1
    Ps, Pm = mvhe.getinvertiblematrix(dim + T_cols)
    T = mvhe.getRandomMatrix(dim, T_cols, mvhe.tBound)
    S = mvhe.getSecretKey(T, Ps)
    #加密
    encOftrainDataReshape = np.zeros((trainDataNums, dim + T.shape[1]), dtype=object)
    for i in range(trainDataNums):
        encOftrainDataReshape[i] = mvhe.encrypt(T, Pm, trainDataReshape[i])
    return encOftrainDataReshape, S, T_cols


def QueryGen(queryPoint,S, T_cols):
    """
    用于用户输入的一个查询数据点，产生线性变换所需矩阵
    :param queryPoint:
    :return:
    """
    #对用户输入的queryPoint做特殊构造以便于后文的计算
    # print(queryPoint.shape)
    # print( queryPoint[0])
    dim = queryPoint.shape[0]   #注意数据维度的获取
    #对查询向量做特殊构造
    newQueryPoint = np.zeros((1, dim + 2), dtype=object)
    temp = queryPoint.dot(queryPoint.T)
    newQueryPoint[0][0] = temp
    newQueryPoint[0][1] = 1
    for j in range(dim):
        newQueryPoint[0][j + 2] = (-2) * queryPoint[j]
    dim = newQueryPoint.shape[1]    #更新dim
    newQueryPoint.resize((1, dim))  # 转换数据维度
    G = (np.copy(newQueryPoint))
    newPs, newPm = mvhe.getinvertiblematrix(1 + T_cols)
    newT = mvhe.getRandomMatrix(1, T_cols, mvhe.tBound)  # 产生新随机矩阵T
    newS = mvhe.getSecretKey(newT, newPs)  # 新密钥
    GS = G.dot(S)
    M = mvhe.KeySwicthMatrix(GS, newT, newPm)
    return M,newS


def ScoreCalculate(trainDataOfEnc, M ,T_cols):
    encDist = np.zeros((trainDataNums, 1 + T_cols),dtype=object)  #这里存疑，需要告知云端T_cols吗
    for i in range(trainDataNums):
        temp = trainDataOfEnc[i]
        encDist[i] = (mvhe.KeySwitch(M, temp))
    return encDist   # send the encDist to the user

def SCMC(encDist, newS, kPoints=3):
    numPoints = encDist.shape[0]
    decDist = np.zeros((numPoints, 1),dtype=object)
    for i in range(numPoints):
        decDist[i][0] = mvhe.decrypt(newS, encDist[i])
    return knn(decDist, kPoints)

def knn(decDist, kPoints):
    numPoints = decDist.shape[0]
    all_dist = np.zeros((numPoints, 2))  # 存样本序号 和 对应距离
    for i in range(numPoints):
        all_dist[i][0] = i
        all_dist[i][1] = decDist[i][0]
    sort_dist = sorted(all_dist, key=lambda all_dist: all_dist[1])
    k_dist = getK(sort_dist, kPoints)
    class_1 = 0
    class_0 = 0
    for x in range(kPoints):
        # num = k_dist[x][0]
        num = int(k_dist[x][0])  # 获取数据对应的序号，目的是为了获得对应的标签
        if (trainData_Tag[num] == 1):
            class_1 += 1
        else:
            class_0 += 1
    # 根据 “得票数” 对预测数进行分类
    if class_1 > class_0:
        labelPre = 1
    else:
        labelPre = 0
    return labelPre





def AccRate(testDataLables, testDataLabelsPredicts ):
    accNum = 0
    for i in range( testDataLables.shape[0]):
        # print(testDataLabels - testDataLabelsPredicts)
        if (testDataLables - testDataLabelsPredicts)[i][0] == 0:
            accNum += 1
    print( "the predict correct rate is %.2f %%" %((accNum / testDataLables.shape[0] )* 100) )





if __name__ == '__main__':
    timeStar = clock()
    print( type(trainData[0][0]))
    trainDataOfEnc, S,T_cols = DataUpload(trainData)
    testPoints = testData.shape[0]
    global labelPreMat
    labelPreMat = np.zeros((testPoints, 1), dtype=object)
    for i in range( testPoints):
        queryPoint = np.copy( testData[i] )
        M, newS = QueryGen(queryPoint, S,T_cols)
        encDist = ScoreCalculate( trainDataOfEnc, M, T_cols)
        labelPre = SCMC(encDist, newS, kPoints=7)
        labelPreMat[i][0] = labelPre
    timeEnd = clock()
    AccRate(testData_Tag, labelPreMat)
    print("the time cost:%.2fms" % ((timeEnd - timeStar) * 1000))












































