import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'      # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛法求解二维泊松方程。
    
    参数:
        M (int): 网格每边的点数
        target (float): 收敛容差
        max_iterations (int): 最大迭代次数
    
    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛
    """
    h = 1.0
    phi = np.zeros((M+1, M+1), dtype=float)
    phi_prev = np.copy(phi)
    rho = np.zeros((M+1, M+1), dtype=float)
    
    # 设置正负电荷位置
    pos_y1, pos_y2 = int(0.6*M), int(0.8*M)
    pos_x1, pos_x2 = int(0.2*M), int(0.4*M)
    neg_y1, neg_y2 = int(0.2*M), int(0.4*M)
    neg_x1, neg_x2 = int(0.6*M), int(0.8*M)
    
    rho[pos_y1:pos_y2, pos_x1:pos_x2] = 1.0
    rho[neg_y1:neg_y2, neg_x1:neg_x2] = -1.0
    
    delta = 1.0
    iterations = 0
    converged = False
    
    while delta > target and iterations < max_iterations:
        phi[1:-1, 1:-1] = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] +
                                  phi[1:-1, :-2] + phi[1:-1, 2:] +
                                  h*h * rho[1:-1, 1:-1])
        delta = np.max(np.abs(phi - phi_prev))
        phi_prev = np.copy(phi)
        iterations += 1
    
    converged = bool(delta <= target)
    
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    可视化电势分布图。
    
    参数:
        phi (np.ndarray): 电势数组
        M (int): 网格大小
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower',
                    cmap='RdBu_r', interpolation='bilinear')
    cbar = plt.colorbar(im)
    cbar.set_label('电势值 (V)', fontsize=12)
    
    plt.fill_between([20, 40], [60, 60], [80, 80], alpha=0.3, color='red', label='正电荷区域')
    plt.fill_between([60, 80], [20, 20], [40, 40], alpha=0.3, color='blue', label='负电荷区域')
    
    plt.xlabel('x 方向（网格点）', fontsize=12)
    plt.ylabel('y 方向（网格点）', fontsize=12)
    plt.title('电势分布图\n带有正负电荷的泊松方程', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    输出电势解的分析信息（中文）。
    
    参数:
        phi (np.ndarray): 电势分布
        iterations (int): 迭代次数
        converged (bool): 是否收敛
    """
    print(f"解的分析：")
    print(f"  迭代次数：{iterations}")
    print(f"  是否收敛：{converged}")
    print(f"  最大电势值：{np.max(phi):.6f} V")
    print(f"  最小电势值：{np.min(phi):.6f} V")
    print(f"  电势范围：{np.max(phi) - np.min(phi):.6f} V")
    
    max_idx = np.unravel_index(np.argmax(phi), phi.shape)
    min_idx = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"  最大电势位置：({max_idx[0]}, {max_idx[1]})")
    print(f"  最小电势位置：({min_idx[0]}, {min_idx[1]})")

if __name__ == "__main__":
    print("正在使用松弛法求解二维泊松方程...")
    
    M = 100
    target = 1e-6
    max_iter = 10000
    
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    analyze_solution(phi, iterations, converged)
    visualize_solution(phi, M)
    
    # 绘制中心截面电势分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    center_y = M // 2
    plt.plot(phi[center_y, :], 'b-', linewidth=2)
    plt.xlabel('x 方向（网格点）')
    plt.ylabel('电势 (V)')
    plt.title(f'y = {center_y} 处的电势分布')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    center_x = M // 2
    plt.plot(phi[:, center_x], 'r-', linewidth=2)
    plt.xlabel('y 方向（网格点）')
    plt.ylabel('电势 (V)')
    plt.title(f'x = {center_x} 处的电势分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
