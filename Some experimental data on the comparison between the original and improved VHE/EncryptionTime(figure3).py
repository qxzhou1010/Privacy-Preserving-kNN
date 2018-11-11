#coding=utf-8
#Author:Qixian Zhou
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt
import numpy as np

dim = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
M1 = [14, 65, 100, 173, 272, 1094, 2471, 4398, 6953, 10010, 13782, 17468, 22467, 27463]
M2 = [1, 2, 6, 10, 17, 87, 314, 536, 996, 1675, 2601, 4399, 6155, 8575]
# plt.figure( figsize=(8,6))
plt.figure()
plt.xlim(0,500)
plt.ylim(0,30000)
plt.plot( dim, M1,marker = "o",label="The Original VHE")
plt.plot(dim,M2,marker="v", label="The Improved VHE")
plt.xlabel("The length of x")
plt.ylabel("The Encryption Time (ms)")
# plt.title("The encryption time between Original VHE and Improved VHE")
plt.legend(loc = "upper left")
plt.savefig("The Encryption Time (ms)",dpi=1000)
plt.show()


