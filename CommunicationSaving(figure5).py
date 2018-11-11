#coding=utf-8
#Author:Qixian Zhou
#coding=utf-8
#Author:Qixian Zhou
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt
import numpy as np

dim = [ 20,  40, 60,80, 100]
M1 = [0.476190476,
0.488372093,
0.4909625,
0.494117647,
0.495327103
]
M2 = [0.47027027,
0.479452055,
0.490909091,
0.493150685,
0.49726776
]
M3 = [0.471428571,
0.485714286,
0.492890995,
0.494661922,
0.495726496
,
]
M4 = [0.476635514,
0.490654206,
0.49375,
0.494145199,
0.494382022,
]

# plt.figure( figsize=(8,6))
plt.figure()
plt.xlim(0,100)
plt.ylim(0.4,0.5)
plt.plot( dim, M1,marker = "o",label="The dimension of query m=4")
plt.plot(dim,M2,marker="o", label="The dimension of query m=9")
plt.plot(dim,M3,marker="o", label="The dimension of query m=20")
plt.plot(dim,M4,marker="o", label="The dimension of query m=32")
plt.xlabel("Number of query")
plt.ylabel("percentage of Communication cost savings")
# plt.title("The encryption time between Original VHE and Improved VHE")
plt.legend(loc="lower left")
plt.savefig("figure_6.png",dpi=1000)
plt.show()


