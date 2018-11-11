#coding=utf-8
#Author:Qixian Zhou
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt
import numpy as np

dim = [0, 20,  40, 60,80, 100]
M1 = [0,17.76,32.43,45.74,63.96,73.43]
M2 = [0,21.03,41.59,56.21,74.13,90.35]
M3 = [0,28.86,57.31,88.65,104.57,129.92]
M4 = [0,41.1,77.45,105.85,146.94,179.12]
# plt.figure( figsize=(8,6))
plt.figure()
plt.xlim(0,100)
plt.ylim(0,240)
plt.plot( dim, M1,marker = "o",label="The dimension of query m=4")
plt.plot(dim,M2,marker="v", label="The dimension of query m=9")
plt.plot(dim,M3,marker="^", label="The dimension of query m=20")
plt.plot(dim,M4,marker="D", label="The dimension of query m=32")
plt.xlabel("Number of query")
plt.ylabel("Time cost(ms)")
# plt.title("The encryption time between Original VHE and Improved VHE")
plt.legend(loc="upper left")
plt.savefig("figure_5.png",dpi=2000)
plt.show()


