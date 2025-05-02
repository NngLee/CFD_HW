import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(Nx, Ny, omega, epsilon=1e-4):
    T = np.ones((Ny, Nx)) * 20  # 初始化
    T[:, 0] = 100  # 左侧边界条件
    max_iter = 1000
    for _ in range(max_iter):
        max_error = 0
        for j in range(1, Nx-1):
            for i in range(1, Ny-1):
                old = T[i, j]
                T[i, j] = (1-omega)*old + omega*0.25*(T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
                max_error = max(max_error, abs(T[i,j]-old))
        if max_error < epsilon:
            break
    return T, _

# 示例调用：计算151×121网格，ω=1.5
T, iterations = solve_heat_equation(151, 121, 1.5)

# 绘制等温线
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.contour(T, levels=np.arange(20, 101, 5))
plt.title("等温线分布")
plt.show()