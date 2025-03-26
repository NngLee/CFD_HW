import numpy as np
import matplotlib.pyplot as plt

def second_order_derivative_central_difference(f, x, h):
    """
    二阶中心差分法计算二阶导数
    参数：
        f : 目标函数 
        x : 求导点
        h : 步长
    返回：
        二阶导数的近似值
    """
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2  # 核心差分公式

# 测试函数及精确导数
def test_func(x):
    return np.sin(x)  # 原函数（可替换为其他测试函数）

def exact_second_deriv(x):
    return -np.sin(x)  # 精确二阶导数（sin(x)的二阶导数为-sin(x)）

# 验证参数设置
x0 = np.pi/4        # 测试点（π/4）
h0 = 0.1            # 初始步长
n = 10              # 验证次数

# 生成步长序列（等比递减）
h_list = [h0 / (2**i) for i in range(n)]
errors = []

# 进行精度验证
for h in h_list:
    approx = second_order_derivative_central_difference(test_func, x0, h)
    exact = exact_second_deriv(x0)
    errors.append(abs(approx - exact))  # 计算绝对误差

# 绘制收敛曲线
plt.figure(figsize=(8, 5))
plt.loglog(h_list, errors, 'bo-', label='实际误差', markersize=8)  # 对数坐标
plt.plot(h_list, [h**2 for h in h_list], 'r--', label=r"O($h^2$)参考线", linewidth=2)

# 中文显示配置
plt.rcParams.update({
    'font.sans-serif': 'SimHei',   # 中文字体设置
    'mathtext.fontset': 'stix',    # 数学符号字体
    'axes.unicode_minus': False    # 解决负号显示问题
})

# 图表标注
plt.xlabel("步长 h", fontsize=12, fontweight='bold')
plt.ylabel("绝对误差", fontsize=12, fontweight='bold')
plt.title("二阶导数中心差分收敛性验证 (f=sin(x))", fontsize=14, pad=20)

# 辅助元素
plt.legend(loc='best', fontsize=10)  # 自动选择图例位置
plt.grid(True, which='both', linestyle='--', alpha=0.7)  # 双网格线
plt.tight_layout()  # 自动调整布局

# 保存和显示
plt.savefig('second_order_convergence.png', dpi=300, bbox_inches='tight')
plt.show()