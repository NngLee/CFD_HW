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


def FDS_roe(UL, UR):
    # 提取左右状态的物理量
    rhoL, uL, pL = get_value(UL)
    rhoR, uR, pR = get_value(UR)
    hL = (gamma/(gamma-1)) * pL/rhoL + 0.5*uL**2  # 左侧比焓
    hR = (gamma/(gamma-1)) * pR/rhoR + 0.5*uR**2  # 右侧比焓

    sqrt_rhoL = np.sqrt(rhoL)
    sqrt_rhoR = np.sqrt(rhoR)

    # Roe平均
    u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)  # Roe平均速度
    H_roe = (sqrt_rhoL * hL + sqrt_rhoR * hR) / (sqrt_rhoL + sqrt_rhoR)  # Roe平均总焓
    a_roe = np.sqrt((gamma-1)*(H_roe - 0.5*u_roe**2))  # Roe平均声速

    # 计算特征速度
    lambda1 = u_roe - a_roe  # 左行声波特征速度
    lambda2 = u_roe          # 中间熵波特征速度
    lambda3 = u_roe + a_roe  # 右行声波特征速度

    # 熵修正，避免特征值为零
    eps = 1e-6
    lambda1 = np.where(np.abs(lambda1) < eps, (lambda1**2 + eps**2)/(2*eps), lambda1)
    lambda3 = np.where(np.abs(lambda3) < eps, (lambda3**2 + eps**2)/(2*eps), lambda3)

    lambda_abs = np.array([np.abs(lambda1), np.abs(lambda2), np.abs(lambda3)])  # 特征值绝对值

    # 通量差分
    delta_U = UR - UL

    # 右特征向量矩阵
    R = np.zeros((3, 3))
    R[0, :] = [1, 1, 1]
    R[1, :] = [u_roe - a_roe, u_roe, u_roe + a_roe]
    R[2, :] = [H_roe - u_roe*a_roe, 0.5*u_roe**2, H_roe + u_roe*a_roe]

    # 左特征向量矩阵
    b1 = 0.5 * (gamma - 1) * u_roe**2 / a_roe**2
    b2 = (gamma - 1) / a_roe**2

    L = np.zeros((3, 3))
    L[0, :] = [0.5*(b1 + u_roe/a_roe), -0.5*(b2*u_roe + 1/a_roe), 0.5*b2]
    L[1, :] = [1 - b1, b2*u_roe, -b2]
    L[2, :] = [0.5*(b1 - u_roe/a_roe), -0.5*(b2*u_roe - 1/a_roe), 0.5*b2]

    # 波强度
    alpha = L @ delta_U

    # 耗散项
    abs_lambda_alpha = lambda_abs * alpha
    diss_vector = R @ abs_lambda_alpha

    # 左右物理通量
    F_L = np.array([rhoL*uL, rhoL*uL**2 + pL, uL*(UL[2] + pL)])
    F_R = np.array([rhoR*uR, rhoR*uR**2 + pR, uR*(UR[2] + pR)])

    # Roe格式数值通量
    F_roe = 0.5 * (F_L + F_R) - 0.5 * diss_vector

    return F_roe
# Minmod限制器
def minmod(v, limiter='minmod'):
    n = len(v)
    vL = np.zeros_like(v)  # 左界面重构值
    vR = np.zeros_like(v)  # 右界面重构值
    
    # 边界点处理（采用一阶重构，直接取原值）
    vL[0] = v[0]
    vR[0] = v[0]
    vL[-1] = v[-1]
    vR[-1] = v[-1]
    
    for i in range(1, n-1):
        # 计算左右两侧的斜率
        deltaL = v[i] - v[i-1]  # 左侧斜率
        deltaR = v[i+1] - v[i]  # 右侧斜率
        
        # 应用minmod限制器
        if deltaL * deltaR <= 0:
            slope = 0  # 斜率异号，取0，防止产生新极值
        else:
            # 取绝对值较小的斜率，保持单调性
            slope = np.sign(deltaL) * min(abs(deltaL), abs(deltaR))
        
        # 计算左右界面的重构值
        vL[i] = v[i] - 0.5 * slope
        vR[i] = v[i] + 0.5 * slope
    
    return vL, vR

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

        # MUSCL-TVD方法进行变量重构
        UL_recon = np.zeros_like(U)
        UR_recon = np.zeros_like(U)

        # 对每个守恒变量分量分别进行重构
        for i in range(3):
            v = U[i, :]
            vL, vR = minmod(v)
            UL_recon[i, :] = vL
            UR_recon[i, :] = vR

           # 计算通量（每个界面一个通量，共 nx+1 个）
        F = np.zeros((3, num + 1))
        
        # 左边界
        F[:, 0] = FDS_roe(UL_recon[:, 0], UR_recon[:, 0])

        # 右边界
        F[:, -1] = FDS_roe(UL_recon[:, -1], UR_recon[:, -1])

        # 内部界面
        for i in range(1, num):
            left_state = UR_recon[:, i - 1]  # 左单元的右界面值
            right_state = UL_recon[:, i]     # 右单元的左界面值
            F[:, i] = FDS_roe(left_state, right_state)
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
        title=f"FDS-TVD N={num}"
    )

if __name__ == "__main__":
    # 参数
    num = 200               # 网格数
    gamma = 1.4             # 流体绝热指数
    t_total = 0.5           # 模拟总时长
    CFL = 0.8               # CFL数
    xmin, xmax = -2, 2      # 计算域
    main()
