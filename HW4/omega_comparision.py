import numpy as np
import matplotlib.pyplot as plt

def solve_sor(Nx, Ny, omega, eps=1e-2, max_iter=10000):
    """
    用SOR方法求解二维稳态热传导问题
    参数:
        Nx, Ny : x和y方向的网格数（包含边界）
        omega  : 松弛因子
        eps    : 收敛判据（最大温度变化阈值）
        max_iter: 最大迭代次数
    返回:
        T : 温度场矩阵
        iterations : 实际迭代次数
    """
    # 初始化温度场（全设为20℃）
    T = np.ones((Ny, Nx)) * 20.0
    
    # 设置边界条件
    T[:, 0] = 100.0    # 左侧边界: 100℃
    T[0, :] = 20.0     # 上侧边界: 20℃
    T[-1, :] = 20.0    # 下侧边界: 20℃
    T[:, -1] = 20.0    # 右侧边界: 20℃
    
    # 迭代计算
    for iterations in range(1, max_iter+1):
        max_error = 0.0
        # 遍历所有内部节点（从1到N-2）
        for j in range(1, Nx-1):      # x方向（列）
            for i in range(1, Ny-1):  # y方向（行）
                old_T = T[i, j]
                # SOR迭代公式（使用已更新的邻居值）
                new_T = (1 - omega) * old_T + omega * 0.25 * (
                    T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1]
                )
                T[i, j] = new_T
                max_error = max(max_error, abs(new_T - old_T))
        
        # 检查收敛条件
        if max_error < eps:
            break
    
    return T, iterations

# 测试不同松弛因子的收敛速度
omegas = [1.0, 1.2, 1.5, 1.7, 1.9]
iterations_list = []

for omega in omegas:
    _, iterations = solve_sor(Nx=16, Ny=13, omega=omega, eps=1e-2)
    iterations_list.append(iterations)
    print(f"ω={omega}: {iterations}次迭代")

# 绘制收敛曲线
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(8, 4))
plt.plot(omegas, iterations_list, 'bo-', markersize=8)
plt.xlabel("Relaxation Factor (ω)", fontsize=12)
plt.ylabel("Iterations to Converge", fontsize=12)
plt.title("Convergence Speed vs. ω (16×13 Grid)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(omegas)
plt.show()