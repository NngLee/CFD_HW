# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import newton
from Sod_exact import sod_exact
from visualize import plot_results

# 初始条件
def initialize(x):
    rho = np.where(x < 0, 1.0, 0.125)   # 密度
    u = np.zeros_like(x)                # 速度
    p = np.where(x < 0, 1.0, 0.1)       # 压力
    return rho, u, p
# 获取三个物理量的值
def get_value(U):
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1) * (U[2] - 0.5 * rho * u**2)
    return rho, u, p


# Van Leer通量向量分裂
def FVS_Van_Leer(U):

    rho = U[0, :]                                           #密度
    u = U[1, :] / rho                                       #速度
    p = (gamma - 1) * (U[2, :] - 0.5 * rho * u**2)          #压强
    a = np.sqrt(gamma * p / rho)                            #声速
    M = u / a                                               #马赫数
    
    # 总通量
    F = np.zeros_like(U)
    F[0, :] = rho * u
    F[1, :] = rho * u**2 + p
    F[2, :] = u * (U[2, :] + p)  

    F_plus = np.zeros_like(U)
    F_minus = np.zeros_like(U)
    
    # Van_Leer分裂
    for i in range(len(rho)):

        if M[i] >= 1:
            F_plus[:, i] = F[:, i]
            F_minus[:, i] = 0.0
        
        elif abs(M[i]) < 1:
            # 正通量分量
            factor_plus = rho[i] * a[i] * (M[i] + 1)**2 / 4.0
            F_plus[0, i] = factor_plus
            F_plus[1, i] = factor_plus * (2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)
            F_plus[2, i] = factor_plus * ((2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)**2 / (2 * (gamma - 1)))
            
            # 负通量分量
            factor_minus = -rho[i] * a[i] * (M[i] - 1)**2 / 4.0
            F_minus[0, i] = factor_minus
            F_minus[1, i] = factor_minus * (-2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)
            F_minus[2, i] = factor_minus * ((-2 * a[i] / gamma + u[i] * (gamma - 1) / gamma)**2 / (2 * (gamma - 1)))
        
        else:  
            F_plus[:, i] = 0.0
            F_minus[:, i] = F[:, i]
    return F_plus, F_minus

def GVC(v, direction='plus'):
    n = len(v)
    v_recon = np.zeros_like(v)
    eps = 1e-12  # 防止除零的小常数
    
    for i in range(n):
        # 边界处理（简单的一阶外推）
        if i == 0:
            # 左边界处理
            if n > 1:
                v_recon[i] = v[i] - 0.5 * (v[i+1] - v[i])
            else:
                v_recon[i] = v[i]
            continue
        elif i == n-1:
            # 右边界处理
            if n > 1:
                v_recon[i] = v[i] + 0.5 * (v[i] - v[i-1])
            else:
                v_recon[i] = v[i]
            continue
        
        # 计算梯度比 r
        if direction == 'plus':
            # 右界面重构 (i+1/2)
            numerator = v[i] - v[i-1]
            denominator = v[i+1] - v[i]
        else:  # direction == 'minus'
            # 左界面重构 (i-1/2)
            numerator = v[i] - v[i+1]
            denominator = v[i-1] - v[i]
        
        # 防止除零
        if np.abs(denominator) < eps:
            r = 0.0
        else:
            r = numerator / denominator
        
        # GVC限制器函数
        if np.abs(r) > 1:
            phi = 1.0
        else:
            phi = r
        
        # 重构界面值
        if direction == 'plus':
            v_recon[i] = v[i] + phi * (v[i+1] - v[i]) / 2
        else:  # direction == 'minus'
            v_recon[i] = v[i] - phi * (v[i] - v[i-1]) / 2
    
    return v_recon

# 三阶Runge-Kutta时间推进
def RK3_step(U, dt, f):
    k1 = f(U)
    k2 = f(U + dt * k1)
    k3 = f(U + dt * (0.25*k1 + 0.25*k2))
    U_new = U + dt * (1/6*k1 + 1/6*k2 + 2/3*k3)
    return U_new

def main():

    # 创建空间网格
    x = np.linspace(xmin, xmax, num)
    dx = (xmax - xmin) / (num - 1)

    # 初始化变量
    t = 0
    iteration = 0
    rho, u, p = initialize(x)
    # 构造守恒变量数组 U
    U = np.array([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u**2])  
    while t < t_total:
        # 计算当前时间步的最大波速和时间步长
        rho, u, p = get_value(U)
        c = np.sqrt(gamma * p / rho)  # 计算声速
        max_speed = np.max(np.abs(u) + c)  # 计算最大特征速度
        dt = CFL * dx / max_speed
        dt = min(dt, t_total - t)  # 保证最后一步不超过总时间

        # 使用GVC重构
        UL_recon = np.zeros_like(U)
        UR_recon = np.zeros_like(U)
        
        # 对每个守恒变量分量分别进行GVC重构
        for comp in range(3):
            v = U[comp, :]
            
            # 重构左界面值 (i-1/2) - 使用minus方向
            vL = GVC(v, direction='minus')
            
            # 重构右界面值 (i+1/2) - 使用plus方向
            vR = GVC(v, direction='plus')

            UL_recon[comp, :] = vL
            UR_recon[comp, :] = vR
        
        # 计算通量（每个界面一个通量，共 nx+1 个）
        F = np.zeros((3, num + 1))
        
        # 内部界面
        for i in range(1, num):
            # 左单元右界面状态
            UL = UR_recon[:, i-1]
            # 右单元左界面状态
            UR = UL_recon[:, i]
            
            # 计算左右状态的通量分裂
            F_plus_L, _ = FVS_Van_Leer(UL.reshape(3, 1))
            _, F_minus_R = FVS_Van_Leer(UR.reshape(3, 1))

            # FVS通量 = F^+(左) + F^-(右)
            F[:, i] = F_plus_L.flatten() + F_minus_R.flatten()
        
        # 边界处理
        # 左边界 (i=0)
        F_plus_L, _ = FVS_Van_Leer(U[:, 0].reshape(3, 1))
        _, F_minus_R = FVS_Van_Leer(U[:, 0].reshape(3, 1))
        F[:, 0] = F_plus_L.flatten() + F_minus_R.flatten()
        
        # 右边界 (i=nx)
        F_plus_L, _ = FVS_Van_Leer(U[:, -1].reshape(3, 1))
        _, F_minus_R = FVS_Van_Leer(U[:, -1].reshape(3, 1))
        F[:, -1] = F_plus_L.flatten() + F_minus_R.flatten()
    

        # 计算通量差分
        dF = (F[:, 1:] - F[:, :-1]) / dx

        # 使用三阶Runge-Kutta方法推进时间
        U = RK3_step(U, dt, lambda U: -dF)

        # 更新时间和迭代次数
        t += dt
        iteration += 1
        print(f"Iteration: {iteration}, Time: {t:.4f}, dt: {dt:.6f}")

    # 计算Sod激波管问题的精确解
    x_exact = np.linspace(xmin, xmax, 1000)
    exact_rho, exact_u, exact_p = sod_exact(x_exact, t_total)

    # 从守恒变量还原物理量
    rho_num = U[0, :]
    u_num = U[1, :] / rho_num
    p_num = (gamma - 1) * (U[2, :] - 0.5 * rho_num * u_num**2)

    # 绘制数值解与精确解对比图
    plot_results(
        x, rho_num, u_num, p_num,
        x_exact, exact_rho, exact_u, exact_p,
        title="FVS + TVD"
    )

if __name__ == "__main__":
    # 参数
    num = 200               # 网格数
    gamma = 1.4             # 流体绝热指数
    t_total = 0.5           # 模拟总时长
    CFL = 0.8               # CFL数
    xmin, xmax = -2, 2      # 计算域
    main()
