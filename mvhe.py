#coding=utf-8
#Author:Qixian Zhou
import numpy as np
import random
from time import clock
np.set_printoptions(suppress=True)
np.set_printoptions(precision=30)
#定义全局变量整数w
w = int(10 ** 11)
randBound = 100000      #生成可逆矩阵对时用到的随机数范围
aBound = 1000
tBound = 1000
eBound = 1000
xBound = 10
def getRandomX(dimension):
    x = np.zeros(dimension,dtype=object)
    for i in range(dimension):
        x[i] = random.randint(0,xBound)
    return x

#构造一对可逆矩阵   输入为：矩阵维度（明文维度N+参数k），k是随机矩阵T的列数
def getinvertiblematrix(dimension):
    I1 = np.identity(dimension,dtype=object)
    I2 = np.identity(dimension,dtype=object)
    temp_I1 = np.zeros(dimension,dtype=object)
    for i in range(dimension *5 ):
        s = (np.random.randint(0,randBound) % dimension)     #生成随机数s
        while True:
            d = (np.random.randint(0,randBound) % dimension )
            if d != s:
                break
        o = (np.random.randint(0,randBound) % 3 - 1 )           #o的取值范围：0，-1，1
        temp_I2 = np.copy( I2[s] )                              #记录I2第s行
        for j in range(dimension):                              #记录I1第d列
            temp_I1[j] = I1[j][d]
        if o == 0:                                              #根据o的取值做后续的矩阵变换
            I2[s] = I2[d]                                       #交换I2的第s行和第d行
            I2[d] = temp_I2
            for j in range(dimension):                          #交换I1的第s列和d列
                I1[j][d] = I1[j][s]
                I1[j][s] = temp_I1[j]
        else:
            I2[d] += temp_I2 * o
            for j in range(dimension):
                I1[j][s] += temp_I1[j] * (-o)
    return I1,I2

def getRandomMatrix(rows, cols, Bound):
    A = np.zeros((rows, cols),dtype=object)
    for i in range(rows):
        for j in range(cols):
            A[i][j] = ( random.randint(0,Bound) )    #random.randint 用于产生随机整数，范围：0-Bound
    return A

def getSecretKey(T, St):
    rows = T.shape[0]
    I = np.eye(rows,dtype=object)
    return np.dot ( np.column_stack((I, T)),  St)

def KeySwicthMatrix(S, T, Mt):
    while True:                 #验证矩阵维度是否正确
        if Mt.shape[0] == S.shape[0] + T.shape[1]:
            break
        print("KeySwitcMatrix Error!")
    A = getRandomMatrix(T.shape[1], S.shape[1], aBound )
    return Mt.dot( np.row_stack( (S - T.dot(A), A ) )  )

def KeySwitch(M, c):
    e = np.zeros(M.shape[0],dtype=object)
    for i in range(M.shape[0]):
        e[i] = random.randint(0,eBound)
    return  np.add( np.dot(M,c) , e)

def encrypt(T, Mt, x):
    length = x.shape[0]                  #获取明文向量长度！用于后面生成单位矩阵
    I = np.eye(length,dtype=object)      #生成单位矩阵
    M = KeySwicthMatrix(I, T, Mt)        #生成密钥转换矩阵M
    c = w * x
    return KeySwitch(M, c)

# def decrypt(S,c):
#     while True:
#         if S.shape[1] == c.shape[0]:
#             break
#         print("Decrypt Error!")
#     Sc = np.dot(S,c)
#     # print("Sc=",Sc)
#     len = Sc.shape[0]
#     output = np.zeros(len,dtype=object)
#     for i in range(len):
#         output[i] = np.round( Sc[i] / w  )
#     output.resize((len, 1))
#     return output.T

def decrypt(S,c):
    while True:
        if S.shape[1] == c.shape[0]:
            break
        print("Decrypt Error!")
    Sc = np.dot(S,c)
    len = Sc.shape[0]
    output = np.zeros(len,dtype=object)
    for i in range(len):
        output[i] = nearestInteger(Sc[i], w)
    output.resize((len, 1))
    return output.T

def nearestInteger(x,w):
    return int( ( x + (w + 1)/2 ) / w)

def addVector(c1, c2):
    return c1 + c2

def linearTransformClient(G, S, T, Mt):
    return KeySwicthMatrix( np.dot(G,S), T, Mt)
def linearTransform(M, c):
    # return np.dot(M, c)
    return KeySwitch(M,c)











