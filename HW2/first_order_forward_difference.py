import numpy as np
import matplotlib.pyplot as plt

def first_order_derivative_forward_difference(f, x, h):
    """
    一阶向前差分法计算导数
    参数：
        f : 目标函数 
        x : 求导点
        h : 步长
    返回：
        导数的近似值
    """
    return (f(x + h) - f(x)) / h

# 测试函数及精确导数
def test_func(x):
    return np.sin(x)  # 可替换为其他测试函数

def exact_deriv(x):
    return np.cos(x)  # 对应导数公式

# 验证参数设置
x0 = np.pi/4        # 测试点
h0 = 0.1            # 初始步长
n = 10              # 验证次数

# 生成步长序列和误差存储
h_list = [h0 / (2**i) for i in range(n)]
errors = []

# 进行精度验证
for h in h_list:
    approx = first_order_derivative_forward_difference(test_func, x0, h)
    exact = exact_deriv(x0)
    errors.append(abs(approx - exact))

# 绘制收敛曲线
plt.figure(figsize=(8, 5))
plt.loglog(h_list, errors, 'bo-', label='实际误差')
plt.plot(h_list, [h for h in h_list], 'r--', label=r"O($h$)参考线")

# 中文字体配置
plt.rcParams.update({
    'font.sans-serif': 'SimHei',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})

# 图表标注
plt.xlabel("步长 h", fontsize=12)
plt.ylabel("绝对误差", fontsize=12)
plt.title("一阶向前差分收敛性验证 (f=sin(x))", fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.savefig('forward_convergence.png', dpi=150)
plt.show()