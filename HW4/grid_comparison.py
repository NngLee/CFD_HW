import numpy as np
import matplotlib.pyplot as plt

def solve_sor(Nx, Ny, omega, eps=1e-2, max_iter=10000):
    """
    用SOR方法求解二维稳态热传导问题
    参数说明:
        Nx, Ny : x和y方向的网格节点数（包含边界）
        omega  : 松弛因子
        eps    : 收敛阈值（最大温度变化 < eps时停止）
        max_iter: 最大允许迭代次数
    返回:
        T : 温度场矩阵（Ny行×Nx列）
        iterations : 实际迭代次数
    """
    # 初始化温度场（所有内部节点设为20℃）
    T = np.ones((Ny, Nx)) * 20.0
    
    # 设置固定边界条件
    T[:, 0] = 100.0    # 左侧边（长边15cm）：100℃
    T[0, :] = 20.0     # 上侧边（短边12cm）：20℃
    T[-1, :] = 20.0    # 下侧边：20℃
    T[:, -1] = 20.0    # 右侧边：20℃
    
    # 开始迭代
    for iterations in range(1, max_iter + 1):
        max_error = 0.0
        # 遍历所有内部节点（i: 行索引，j: 列索引）
        for j in range(1, Nx-1):
            for i in range(1, Ny-1):
                old_T = T[i, j]
                # SOR公式：使用最新计算的邻居值（按列顺序更新）
                new_T = (1 - omega) * old_T + omega * 0.25 * (
                    T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1]
                )
                T[i, j] = new_T
                # 计算当前节点误差
                max_error = max(max_error, abs(new_T - old_T))
        
        # 检查是否收敛
        if max_error < eps:
            break
    
    return T, iterations

# 定义不同网格尺度
grids = {
    "Coarse (5×4)": (5, 4),     # 15cm→Δx=3.75cm, 12cm→Δy=4cm
    "Medium (15×12)": (15, 12), # Δx=1cm, Δy=1cm
    "Fine (30×24)": (30, 24)    # Δx=0.5cm, Δy=0.5cm
}

# 寻找每个网格的最佳松弛因子
best_omegas = {}
for grid_name, (Nx, Ny) in grids.items():
    print(f"\n正在测试网格：{grid_name} ({Nx}×{Ny})")
    # 在1.0~1.9之间测试10个ω值
    test_omegas = np.linspace(1.0, 1.9, 10)
    min_iter = float('inf')
    best_omega = 1.0
    
    for omega in test_omegas:
        _, iterations = solve_sor(Nx, Ny, omega, eps=1e-2)
        print(f"  ω={omega:.2f} → 迭代次数={iterations}")
        # 更新最佳ω值
        if iterations < min_iter:
            min_iter = iterations
            best_omega = omega
    
    best_omegas[grid_name] = round(best_omega, 2)  # 保留两位小数

# 输出结果
print("\n最佳松弛因子：")
for grid, omega in best_omegas.items():
    print(f"  {grid}: ω={omega}")

# 绘制结果曲线
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(10, 5))
for grid_name, (Nx, Ny) in grids.items():
    # 获取该网格所有测试ω的迭代次数
    test_omegas = np.linspace(1.0, 1.9, 10)
    iterations_list = []
    for omega in test_omegas:
        _, iterations = solve_sor(Nx, Ny, omega, eps=1e-2)
        iterations_list.append(iterations)
    # 绘制曲线
    plt.plot(test_omegas, iterations_list, 'o-', label=grid_name)

plt.xlabel("Relaxation Factor (ω)", fontsize=12)
plt.ylabel("Iterations to Converge", fontsize=12)
plt.title("不同网格尺度下的最佳松弛因子", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()