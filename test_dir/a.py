import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import seaborn as sns

sns.set_style("darkgrid")


# unitZ = np.random.random(10)
# print(unitZ)
# print(type(unitZ))
N = 1234
unitZ = np.array([np.random.uniform(-1,1) for i in range(N)])
rad = np.array([np.deg2rad(np.random.uniform(0,360)) for i in range(N)])

X = np.sqrt(1 - unitZ * unitZ) * np.cos(rad)
Y = np.sqrt(1 - unitZ * unitZ) * np.sin(rad)
Z = unitZ



fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#.plotで描画
#linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
#markerは無難に丸
ax.scatter(X,Y,Z, s = 10,c = "blue")

plt.show()
# fig.savefig("kadai1.png")import matplotlib.pyplot as plt
