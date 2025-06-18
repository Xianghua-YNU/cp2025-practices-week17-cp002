"""
学生模板：松弛迭代法解常微分方程
文件：relaxation_method_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

def solve_ode(h, g, max_iter=10000, tol=1e-6):
    """
    松弛迭代法求解二阶常微分方程边值问题
    d²x/dt² = -g  with x(0)=0, x(10)=0
    
    参数:
        h: 时间步长 (s)
        g: 重力加速度 (m/s²)
        max_iter: 最大迭代次数
        tol: 收敛容差
        
    返回:
        tuple: (时间数组, 解数组)
    """
    t = np.arange(0, 10 + h, h)
    x = np.zeros_like(t)
    delta = float('inf')
    iteration = 0
    
    # 核心迭代结构
    while delta > tol and iteration < max_iter:
        x_new = np.copy(x)
        x_new[1:-1] = 0.5 * (h**2 * g + x[2:] + x[:-2])
        delta = np.max(np.abs(x_new - x))
        x = x_new
        iteration += 1
    
    return t, x

if __name__ == "__main__":
    h = 0.1
    g = 9.8
    
    t, x_num = solve_ode(h, g)
    x_analytical = -0.5 * g * t**2 + 5 * g * t
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, x_num, 'b-', linewidth=2, label='Numerical Solution')
    plt.plot(t, x_analytical, 'r--', label='Analytical Solution')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title('Projectile Trajectory (Relaxation Method)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()
    
    max_error = np.max(np.abs(x_num - x_analytical))
    print(f"Maximum absolute error: {max_error:.3e}")
