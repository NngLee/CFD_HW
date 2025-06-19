import numpy as np
from scipy.optimize import newton
def sod_exact(x, t):
    # 初始左右状态
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    gamma = 1.4
    
    # 定义中间压力的Newton迭代函数
    def f(p_star):
        if p_star > pL:
            # 左激波
            AL = (p_star - pL) * np.sqrt((1 - (gamma-1)/(gamma+1)) / (rhoL * (p_star + (gamma-1)/(gamma+1)*pL)))
        else:
            # 左膨胀波
            AL = (2*aL)/(gamma-1) * ((p_star/pL)**((gamma-1)/(2*gamma)) - 1)
        
        if p_star > pR:
            # 右激波
            AR = (p_star - pR) * np.sqrt((1 - (gamma-1)/(gamma+1)) / (rhoR * (p_star + (gamma-1)/(gamma+1)*pR)))
        else:
            # 右膨胀波
            AR = (2*aR)/(gamma-1) * ((p_star/pR)**((gamma-1)/(2*gamma)) - 1)
        
        return AL + AR - (uR - uL)
    
    # 计算声速
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    
    # 使用牛顿迭代法求解中间压力
    p_initial = 0.5 * (pL + pR)
    try:
        p_star = newton(f, p_initial, maxiter=100, tol=1e-6)
    except:
        p_star = p_initial  # 若迭代失败，使用初始猜测
    
    # 计算中间速度
    if p_star > pL:
        # 左激波
        u_star = uL + (p_star - pL) * np.sqrt((1 - (gamma-1)/(gamma+1)) / (rhoL * (p_star + (gamma-1)/(gamma+1)*pL)))
    else:
        # 左膨胀波
        u_star = uL + (2*aL)/(gamma-1) * (1 - (p_star/pL)**((gamma-1)/(2*gamma)))
    
    # 计算波系结构
    # 左膨胀波尾部速度（如果存在）
    if p_star <= pL:
        a_starL = aL * (p_star/pL)**((gamma-1)/(2*gamma))
        u_exp_left = u_star - a_starL
    else:
        # 左激波速度
        u_shock_left = uL - aL * np.sqrt((gamma+1)/(2*gamma)*(p_star/pL - 1) + 1)
    
    # 右激波速度（如果存在）
    if p_star > pR:
        u_shock_right = uR + aR * np.sqrt((gamma+1)/(2*gamma)*(p_star/pR - 1) + 1)
    
    # 各区间的解
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    
    for i in range(len(x)):
        xi = x[i] / t if t != 0 else 0  # 避免除以零
        
        # 1. 左均匀区
        if p_star > pL:  # 左激波情况
            if xi <= u_shock_left:
                rho[i] = rhoL
                u[i] = uL
                p[i] = pL
            # 2. 激波后区域
            elif xi <= u_star:
                rho[i] = rhoL * ((gamma+1)*p_star + (gamma-1)*pL) / ((gamma-1)*p_star + (gamma+1)*pL)
                u[i] = u_star
                p[i] = p_star
        else:  # 左膨胀波情况
            # 1. 左均匀区
            if xi <= uL - aL:
                rho[i] = rhoL
                u[i] = uL
                p[i] = pL
            # 2. 膨胀波区
            elif xi <= u_exp_left:
                u[i] = (2/(gamma+1)) * (aL + (gamma-1)/2*uL + xi)
                a = aL - (gamma-1)/2*(u[i] - uL)
                rho[i] = rhoL * (a/aL)**(2/(gamma-1))
                p[i] = pL * (rho[i]/rhoL)**gamma
            # 3. 中间左均匀区
            elif xi <= u_star:
                rho[i] = rhoL * (p_star/pL)**(1/gamma)
                u[i] = u_star
                p[i] = p_star
        
        # 4. 中间右均匀区（接触间断右侧）
        if xi > u_star:
            if p_star > pR:  # 右激波情况
                # 4. 中间右均匀区（激波前）
                if xi <= u_shock_right:
                    rho[i] = rhoR * ((gamma+1)*p_star + (gamma-1)*pR) / ((gamma-1)*p_star + (gamma+1)*pR)
                    u[i] = u_star
                    p[i] = p_star
                # 5. 右均匀区
                else:
                    rho[i] = rhoR
                    u[i] = uR
                    p[i] = pR
            else:  # 右膨胀波情况
                # 4. 中间右均匀区（膨胀波前）
                a_starR = aR * (p_star/pR)**((gamma-1)/(2*gamma))
                u_exp_right = u_star + a_starR
                if xi <= u_exp_right:
                    rho[i] = rhoR * (p_star/pR)**(1/gamma)
                    u[i] = u_star
                    p[i] = p_star
                # 5. 膨胀波区
                elif xi <= uR + aR:
                    u[i] = (2/(gamma+1)) * (-aR + (gamma-1)/2*uR + xi)
                    a = aR + (gamma-1)/2*(u[i] - uR)
                    rho[i] = rhoR * (a/aR)**(2/(gamma-1))
                    p[i] = pR * (rho[i]/rhoR)**gamma
                # 6. 右均匀区
                else:
                    rho[i] = rhoR
                    u[i] = uR
                    p[i] = pR
    
    return rho, u, p

