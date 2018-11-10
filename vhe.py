#author Qixian Zhou
#coding=utf-8
#------------------从文件中读入数据、进行加密解密的验证---------------------------
#------------------改动：解密函数里的返回值经过了转置-----------------------------
#------------------经过验证、加解密是没有问题的-----------
#-----------------转置的目的是  解密后和明文相减不出错---------


import numpy as np
import random
from time import clock

#定义全局变量、重要参数l
w = int(10 ** 11)
#l 对解密当然也有影响啦 ，规定的密文比特化 每一个密文的位数和最大值 w * x < 2^l
#那么  l 其实就主要受我 w大小 的影响
l = 60
aBound = 1000
eBound = 1000
tBound = 1000
xBound = 10


#产生随机矩阵,大小：rows x cols, 数值范围：0-Bound的整数
#行数 列数  数值范围
def getRandomMatrix(rows, cols, Bound):
    A = np.zeros((rows, cols),dtype=object)
    for i in range(rows):
        for j in range(cols):
            A[i][j] = ( random.randint(0,Bound) )    #random.randint 用于产生随机整数，范围：0-Bound
    return A


#生成随机向量
# 向量长度  数值范围
def getRandomX(len):
    A = np.zeros(len,dtype=object)
    for i in range(len):
        A[i] = int(random.randint(0,xBound))
    A.resize((len, 1))
    return A

def getRandomXForMeasure(len):
    A = np.zeros(len,dtype=object)
    for i in range(len):
        A[i] = int(random.randint(0,xBound))
    # A.resize((len, 1))
    return A




#return S*
# S
def get_S_star(S):
    #求出S 的行数和列数
    rows, cols = S.shape
    #定义变量 接收S_star   #注意矩阵大小  #l 是重要参数！
    result = np.zeros((rows, l * cols),dtype=object)
    #定义2^l次放
    # l行 1列
    powers = np.zeros((l,1),dtype=object)
    powers[0][0] = 1
    for i in range(l-1):
        powers[i+1][0] = powers[i][0] * 2
    for i in range(rows):
        for j in range(cols):
            for k in range(l):
                result[i][j*l + k] = S[i][j] * powers[k][0]
    return result

# 返回M
#-----------keySwitch:the core of the vhe------密文的生成: M * （w * x）

def KeySwicthMatrix(S, T):
    #求S*,为求M　做准备
    S_star = get_S_star(S)
    # print("typr S_star", type(S_star[0][0]))
    #获取S 和 T 的行数和列数  为后面随机矩阵 A E的生成做准备！
    S_star_rows, S_star_cols = S_star.shape
    T_rows, T_cols = T.shape
    A = getRandomMatrix(T_cols, S_star_cols, aBound )
    E = getRandomMatrix(S_star_rows, S_star_cols, eBound )
    return (np.row_stack((S_star + E - T.dot(A), A)))

#这里的二进制转化是为了密文比特化做准备的！
#和S*保持一致 结果  “从小到大”
#而且这里的 bit化后的位数要和 l 保持一致！！
def ten_to_2(num):
    num = int(abs(num))
    res=[]
    while True:
        y = num % 2
        num = int(num/2)
        res.append(y)
        if num == 0:
            break
    #下面的代码 来是实现比特化后的数据位数等于 l
    #但是，如果我len(res) > l  呢？？
    if len(res) == l:
        return res
    else:
        for i in range(len(res), l):
            res.append(0)
        return res

#密文比特化
#就是求 C*
def getBitVector(c):
    #获取密文长度
    len = c.shape[0]
    #接收比特化的密文
    c_star = np.zeros((len * l), dtype=object)
    # c_star = []
    for i in range(len):
        if (c[i] < 0):
            sign = -1
        else:
            sign = 1
        #对每一个元素进行bit化
        res = ten_to_2(c[i])
        for j in range(0,l):
            c_star[i*l + j] = sign * res[j]
            # c_star.append(sign * res[j])
    # print( "cstar",c_star.shape)
    # print( type(c_star[0]))
    return c_star

# #密文转化
# #密钥转换矩阵 M  密文c
# #返回 M * C*
# 这个其实就是产生 “新”密文的过程！
def KeySwitch(M, c):
    c_star = getBitVector(c)
    # print("typr cstar", type(c_star[0]))
    # print("c_star",c_star)
    # print("c_star",c_star.shape)
    # c_star.resize(c.shape[1] * l, 1)
    # print( )
    # temp = np.dot(M,c_star)
    # print( "temp", type(temp))
    # print( "temp", type( temp[0]))
    return np.dot(M, c_star)


#定义加密函数
# 随机矩阵T, 明文向量
def encrypt(T, x):
    #获取明文向量长度！用于后面生成单位矩阵
    length = x.shape[0]
    # print( "length%d"%length)
    #生成单位矩阵
    I = np.eye(length, dtype=object)
    # print("enc中I的类型:",type(I[0][0]))
    #生成密钥转换矩阵M
    M = KeySwicthMatrix(I, T)
    # print("typr M", type(M[0][0]))
    #M大小正确！
    # print("M",M.shape)
    c = w * x
    # print("c=",c)
    # print( "type C", type(c[0]))
    return  KeySwitch(M, c)

#定义解密函数
#密钥S 密文c
# def decrypt(S,c):
#     Sc = np.dot(S,c)
#     len = Sc.shape[0]
#     output = np.zeros(len,dtype=int)
#
#     for i in range(len):
#         output[i] = np.round( Sc[i] /w  )   #就近取整
#         # output[i] = np.round( (Sc[i] + 0.5 * (w + 1) ) /w )
#         # output[i] = np.ceil((Sc[i] + 0.5 * (w + 1)) / w)
#     output.resize(len, 1)
#     # return output
#     return output.T

def decrypt(S,c):
    # while True:
    #     if S.shape[1] == c.shape[0]:
    #         break
    #     print("Decrypt Error!")
    Sc = np.dot(S,c)
    # print("Sc", type(Sc))
    # print("Sc",Sc.shape)
    # print("Sc", type(Sc[0]))
    len = Sc.shape[0]
    output = np.zeros(len,dtype=object)
    for i in range(len):
        output[i] = nearestInteger(Sc[i], w)
    # print("output", type(output))
    # print("output", output.shape)
    # print("output", type(output[0]))
    output.resize((len, 1))
    # print("output", output.shape)
    return output.T

def nearestInteger(x,w):
    return int( ( x + (w + 1)/2 ) / w)


#生成密钥S
def getSecretKey(T):
    rows = T.shape[0]
    I = np.eye(rows, dtype=object)

    S = np.column_stack((I, T))
    return S

#生成线性变换矩阵M
def linerTransformClient(G, S, T):
    return KeySwicthMatrix( np.dot(G, S), T)

#线性变化 生成新密文
def linearTransform(M, c):
    c_star = getBitVector(c)
    return np.dot(M, c_star)

#定义随机矩阵的列数、这个列数决定了加密后数据增加的维数、一般取1。


# T_col = 1
# x = np.loadtxt("D_train.txt")
# rows, cols = x.shape
# N = cols
# T = getRandomMatrix(N, T_col, aBound)
# S = getSecretKey(T)
# star = clock()
#
#
#
# #-------------------------对向量进行加密---------------------
# enc_x = np.zeros((rows, cols + 1))
# for i in range(rows):
#     enc_x[i] = encrypt(T, x[i])
# print("enc_x=",enc_x)
# print("enc_x.shape", enc_x.shape)
#
#
#
#
#
#
# #-------------------------解密-----------------------
# # dec_x = np.zeros((rows, cols))
# dec_x = []
# for i in range(rows):
#     temp = decrypt(S, enc_x[i])
#     # print("temp=",temp)
#     # print("temp.shape",temp.shape)
#     dec_x.append(temp)
# # #------------------------解密后的数据和明文对比---------
# end = clock()
# print("time is " + str(end - star) + "s")
# for i in range(rows):
#     print("dec_x - x",dec_x[i] - x[i] )
#     # print("dec_x.shape", dec_x.shape)
#
#
#
#
# #-------------------解密正确！！！------------
# #-------------------解密结束分割线！！！！！！--------------------













