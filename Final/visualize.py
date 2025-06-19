from matplotlib import pyplot as plt
def plot_results(x, rho_num, u_num, p_num, x_exact, exact_rho, exact_u, exact_p, title="FVS+TVD"):
    plt.figure(figsize=(18, 12))
    plt.suptitle(title, fontsize=24, fontweight='bold')

    # 密度图
    plt.subplot(311)
    plt.plot(x, rho_num, 'b-', linewidth=2, label='Numerical')
    plt.plot(x_exact, exact_rho, 'r--', linewidth=2, label='Exact')
    plt.ylabel('Density', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(alpha=0.3)
    plt.title('Density Comparison', fontsize=18)

    # 速度图
    plt.subplot(312)
    plt.plot(x, u_num, 'b-', linewidth=2, label='Numerical')
    plt.plot(x_exact, exact_u, 'r--', linewidth=2, label='Exact')
    plt.ylabel('Velocity', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(alpha=0.3)
    plt.title('Velocity Comparison', fontsize=18)

    # 压力图
    plt.subplot(313)
    plt.plot(x, p_num, 'b-', linewidth=2, label='Numerical')
    plt.plot(x_exact, exact_p, 'r--', linewidth=2, label='Exact')
    plt.ylabel('Pressure', fontsize=16)
    plt.xlabel('x', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.grid(alpha=0.3)
    plt.title('Pressure Comparison', fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
