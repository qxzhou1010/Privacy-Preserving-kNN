#coding=utf-8
#Author:Qixian Zhou
"""
循环，单个点到所有点欧式距离的计算
然后分类预测
"""
import numpy as np
import mvhe0
from time import clock
import time
import random
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
# train_data = np.loadtxt("D_train.txt",dtype=np.int64)
# trainDataNums, dim = train_data.shape
# trainData = np.zeros( (trainDataNums, dim), dtype=object)   #主要想把所有的数据类型都转换为[class:int]
# for i in range(trainDataNums):
#     trainData[i] = np.copy( train_data[i])
# test_data = np.loadtxt("D_test.txt",dtype=np.int64)
# testDataNums, dim = test_data.shape
# testData = np.zeros( (testDataNums, dim), dtype=object)
# for i in range(testDataNums):
#     testData[i] = np.copy( test_data[i])
# trainData_Tag = np.loadtxt( "D_train_result.txt")
# testData_Tag = np.loadtxt( "D_test_result.txt")
# testData_Tag.resize( (testDataNums,1))

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

    #加密
    #   初始化参数
    T_cols = 1
    Ps, Pm = mvhe0.getinvertiblematrix(dim + T_cols)
    T = mvhe0.getRandomMatrix(dim, T_cols, mvhe0.tBound)
    S = mvhe0.getSecretKey(T, Ps)
    I = np.eye(dim, dtype=object)
    A = mvhe0.getRandomMatrix(T.shape[1], I.shape[1], mvhe0.aBound)
    M = mvhe0.SpecialKeySwitchMatrix(I, T, Pm,A)
    # e = np.zeros(M.shape[0], dtype=object)
    # for i in range(M.shape[0]):
    #     e[i] = random.randint(0, mvhe0.eBound)

    #加密
    encOftrainData = np.zeros((trainDataNums, dim + T.shape[1]), dtype=object)
    for i in range(trainDataNums):
        encOftrainData[i] = mvhe0.newSpecialEncrypt(M,trainData[i])
    M = np.array(M, dtype=np.float64)
    MPinv = np.linalg.pinv(M)
    HA = I.dot(MPinv)
    H = HA.T.dot(HA)
    M = mvhe0.SpecialKeySwitchMatrix(I, T, Pm, A)
    return encOftrainData,M,H



    # return encOftrainDataReshape, S, T_cols


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
    newPs, newPm = mvhe0.getinvertiblematrix(1 + T_cols)
    newT = mvhe0.getRandomMatrix(1, T_cols, mvhe0.tBound)  # 产生新随机矩阵T
    newS = mvhe0.getSecretKey(newT, newPs)  # 新密钥
    GS = G.dot(S)
    M = mvhe0.KeySwicthMatrix(GS, newT, newPm)
    return M,newS


def ScoreCalculate(trainDataOfEnc, M ,T_cols):
    encDist = np.zeros((trainDataNums, 1 + T_cols),dtype=object)  #这里存疑，需要告知云端T_cols吗
    for i in range(trainDataNums):
        temp = trainDataOfEnc[i]
        encDist[i] = (mvhe0.KeySwitch(M, temp))
    return encDist   # send the encDist to the user

def SCMC(encDist, newS, kPoints=3):
    numPoints = encDist.shape[0]
    decDist = np.zeros((numPoints, 1),dtype=object)
    for i in range(numPoints):
        decDist[i][0] = mvhe0.decrypt(newS, encDist[i])
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



def userEnc( M,queryPoint):
    return mvhe0.newSpecialEncrypt(M,queryPoint)

def cloudCalcuate(trainDataOfEnc, queryPointEnc, H, kPoints=3):
    trainNums,dim = trainDataOfEnc.shape
    # temp = trainDataOfEnc[0]
    distEncH = np.zeros( (trainDataNums,1),dtype=object)
    for i in range(trainNums):
        temp = trainDataOfEnc[i] - queryPointEnc
        temp.resize( (1,dim))
        tempValue = temp.dot(H)
        distEncH[i][0] = tempValue.dot( temp.T)
    # print( distEncH)
    return knn(distEncH,kPoints)





if __name__ == '__main__':
    timeStar = clock()
    trainDataOfEnc, M, H = DataUpload(trainData)
    testNums = testData.shape[0]
    labelPreMat = np.zeros( (testNums,1),dtype=object)
    encTestData = np.zeros( (testNums,testData.shape[1] + 1), dtype=object)
    for i in range(testNums):
        encTestData[i] = userEnc(M, testData[i])

    for i in range( testNums):
        labelPre = cloudCalcuate(trainDataOfEnc, encTestData[i], H, kPoints=3)
        labelPreMat[i][0] = labelPre
    AccRate(testData_Tag, labelPreMat)
    timeEnd = clock()
    print("the time cost:%.2fms" % ((timeEnd - timeStar) * 1000))


    # trainDataOfEnc, S,T_cols = DataUpload(trainData)
    # testPoints = testData.shape[0]
    # global labelPreMat
    # labelPreMat = np.zeros((testPoints, 1), dtype=object)
    # for i in range( testPoints):
    #     queryPoint = np.copy( testData[i] )
    #     M, newS = QueryGen(queryPoint, S,T_cols)
    #     encDist = ScoreCalculate( trainDataOfEnc, M, T_cols)
    #     labelPre = SCMC(encDist, newS, kPoints=3)
    #     labelPreMat[i][0] = labelPre
    # timeEnd = clock()
    # AccRate(testData_Tag, labelPreMat)
    # print("the time cost:%.2fms" % ((timeEnd - timeStar) * 1000))












































