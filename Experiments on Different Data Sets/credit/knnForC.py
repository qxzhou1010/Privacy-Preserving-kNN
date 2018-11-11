#author Qixian Zhou
import numpy as np
import math

#导入数据集
#训练数据
train_data = np.loadtxt("D_train.txt")
train_tag = np.loadtxt("D_train_result.txt")
#测试数据
test_data = np.loadtxt("D_test.txt")
test_tag = np.loadtxt("D_test_result.txt")
#计算数据集数据个数 维度
train_data_length = train_data.shape[0]
train_data_dimension = train_data.shape[1]
# print(train_data_length)
# print(train_data_dimension)
test_data_length = test_data.shape[0]
test_data_dimension = test_data.shape[1]
# print(test_data_length)
# print(test_data_dimension)
print("train numbers:",train_data_length)

print("test number",test_data_length)
print("the dimension :",test_data_dimension)



#计算两点之间距离 样本1 样本2 样本长度
#计算两个点之间的距离
def Dist(dot_1, dot_2, length):
    distance = 0
    for x in range(length):
        distance += pow((dot_1[x] - dot_2[x]), 2)
    return math.sqrt(distance)


#计算预测数据集中单个点  到  训练数据集中所有点的距离
#并从小到大排序
#计算距离并排序！！
def all_Dist(train_data, test_data):
    all_dist = np.zeros((train_data.shape[0], 2)) #存样本序号 和 对应距离
    for i in range(train_data.shape[0]):
        # test_instance = test_data[i]
        dist = Dist(train_data[i], test_data, train_data.shape[1])
        #记录序号和距离
        all_dist[i][0] = i
        all_dist[i][1] = dist
    sort_dist = sorted(all_dist, key=lambda all_dist: all_dist[1])
    # # k_dist = np.zeros((k))
    # for i in range(k):
    #     k_dist = sort_dist[i]
    return sort_dist

#获取前k个值
def getK(sort_dist, k):
    k_dist = np.zeros((k,2))
    # sort_dist = all_Dist(train_data, test_data[0])
    for i in range(k):
        k_dist[i] = sort_dist[i]
    # print(k_dist)
    return k_dist


#写分类函数
def knn_class(train_data, test_data, k):
    test_pre = np.zeros((test_data.shape[0], 1))  # 用来记录预测分类结果
    for i in range(test_data.shape[0]):
        test_dot = test_data[i]
        sort_dist = all_Dist(train_data, test_dot) #计算记录并排序
        k_dist = getK(sort_dist, k)#获取前k个数据
        #初始化两个变量，用于“投票计数”
        class_1 = 0
        class_0 = 0
        for x in range(k):
            # num = k_dist[x][0]
            num = int(k_dist[x][0]) #获取数据对应的序号，目的是为了获得对应的标签
            if(train_tag[num] == 1):
                class_1 += 1
            else:
                class_0 += 1
        #根据 “得票数” 对预测数进行分类
        if class_1 > class_0:
            test_pre[i] = 1
        else:
            test_pre[i] = 0
    return test_pre

k = 11
test_pre = knn_class(train_data, test_data, k)
# 计算准确率
Acc = 0
for i in range(test_data.shape[0]):
    if test_pre[i] - test_tag[i] == 0:
        Acc += 1
print("the accurancy=" + str((Acc/test_data_length) * 100) + '%' )








