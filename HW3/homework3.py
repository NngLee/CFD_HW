import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

# ======================
# 通用参数设置
# ======================
L = 3.0       # 计算区域长度
T = 2.0       # 总计算时间
c = 1.0       # 波动方程系数

# ======================
# 数值格式实现函数
# ======================
def upwind(u, dx, dt):                                                         #迎风格式
    nu = dt/dx
    return u - nu*(u - np.roll(u, 1))

def lax_friedrichs(u, dx, dt):
    nu = dt/dx
    u_avg = 0.5*(np.roll(u, -1) + np.roll(u, 1))
    return u_avg - 0.5*nu*(np.roll(u, -1) - np.roll(u, 1))                     # Lax_friedrichs格式

def lax_wendroff(u, dx, dt):
    nu = dt/dx
    flux = 0.5*nu*(np.roll(u, -1) - np.roll(u, 1))
    diffusion = 0.5* nu **2 * (np.roll(u, -1) - 2 * u + np.roll(u, 1))         # Lax_wendroff格式
    return u - flux + diffusion 

# ======================
# 精确解函数
# ======================
def exact_solution(x, t):
    return np.sin(2*np.pi*(x - c*t))

# ======================
# 主计算函数
# ======================
def simulate(scheme, dx, cfl):
    dt = cfl*dx/c
    x = np.arange(0, L, dx)
    u = np.sin(2*np.pi*x)
    nt = int(T/dt)
    
    for _ in range(nt):
        u = scheme(u, dx, dt)
    
    return x, u

# ===========================================
# 任务一：稳定性验证（CFL=0.8, 1.0, 1.1）
# ===========================================
plt.figure(figsize=(15,10))
cfl_list = [0.8, 1.0, 1.1]
schemes = [upwind, lax_friedrichs, lax_wendroff]
names = ["Upwind", "Lax-Friedrichs", "Lax-Wendroff"]

for i, scheme in enumerate(schemes):
    for j, cfl in enumerate(cfl_list):
        dx = 0.02
        x, u = simulate(scheme, dx, cfl)
        plt.subplot(3, 3, i*3 + j + 1)
        plt.plot(x, u, label=f"CFL={cfl}")
        plt.plot(x, exact_solution(x, T), '--')
        plt.title(f"{names[i]} (CFL={cfl})")
        plt.ylim(-1.5, 1.5) if cfl > 1 else None

plt.tight_layout()
plt.savefig('stability_comparison.png')
plt.close()

# ===========================================
# 任务二：精度阶数验证
# ===========================================
def convergence_rate(scheme):
    dx_list = [0.1, 0.05, 0.025]
    errors = []
    
    for dx in dx_list:
        x, u_num = simulate(scheme, dx, 0.8)
        u_exact = exact_solution(x, T)
        errors.append(np.linalg.norm(u_num - u_exact)/np.sqrt(len(x)))
    
    def fit_func(x, a, b):
        return a + b*x
    
    popt, _ = curve_fit(fit_func, np.log(dx_list), np.log(errors))
    return popt[1]  # 返回收敛阶

rates = [convergence_rate(scheme) for scheme in schemes]

# ===========================================
# 任务三：耗散与相位分析
# ===========================================
plt.figure(figsize=(12,6))
dx = 0.02
for scheme, name in zip(schemes, names):
    x, u = simulate(scheme, dx, 0.8)
    plt.plot(x, u, label=name)

plt.plot(x, exact_solution(x, T), 'k--', label="Exact")
plt.title("Solution Comparison at t=2")
plt.legend()
plt.savefig('dissipation_phase.png')
plt.close()


# ======================
# 可视化输出
# ======================
plt.figure(figsize=(12,4))
plt.semilogx([0.1,0.05,0.025], [0.1**1, 0.05**1, 0.025**1], '--', label='理论1阶')
plt.semilogx([0.1,0.05,0.025], [0.1**2, 0.05**2, 0.025**2], '--', label='理论2阶')
for scheme, name in zip(schemes, names):
    dx_list = [0.1, 0.05, 0.025]
    errors = []
    for dx in dx_list:
        x, u_num = simulate(scheme, dx, 0.8)
        errors.append(np.linalg.norm(u_num - exact_solution(x, T)))
    plt.semilogx(dx_list, errors, 'o-', label=name)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.xlabel('dx'); plt.ylabel('L2 Error')
plt.legend(); plt.title('收敛阶验证')
plt.savefig('convergence_rate.png')

plt.figure(figsize=(10, 6))
ax = plt.gca()

