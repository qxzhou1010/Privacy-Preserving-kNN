#coding=utf-8
#Author:Qixian Zhou
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt

dim = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
M1 = [0.05, 0.19, 0.43, 0.75, 1.17, 4.62, 10.36, 18.14, 28.72, 41.36, 56.23, 73.42, 92.90, 114.66]
M2 = [0.0008, 0.0032, 0.0071,0.0125, 0.0195, 0.0771, 0.1728, 0.3067, 0.4787, 0.6889, 0.9373, 1.2238, 1.5484,1.9112]


# plt.figure( figsize=(6,6))
plt.figure()
plt.xlim(0,500)
plt.ylim(0,115)
plt.plot( dim, M1,marker = "o",label="The Original VHE")
plt.plot(dim,M2, marker = "v",label="The Improved VHE")
plt.xlabel("The length of x")
plt.ylabel("The size of M(MB)")
# plt.title(" The size of M between Origianl VHE and Improved VHE")
plt.legend()
plt.savefig("The size of M(MB)",dpi=600)
plt.show()


