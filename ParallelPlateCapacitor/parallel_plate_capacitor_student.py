import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'  
matplotlib.rcParams['axes.unicode_minus'] = False




def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    Solve Laplace equation using Jacobi iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        tol (float): Convergence tolerance
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize potential grid
    u = np.zeros((ygrid, xgrid))
    
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    iterations = 0
    max_iter = 10000
    convergence_history = []
    
    while iterations < max_iter:
        u_old = u.copy()
        
        # Jacobi iteration
        u[1:-1,1:-1] = 0.25*(u[2:,1:-1] + u[:-2,1:-1] + u[1:-1, 2:] + u[1:-1,:-2]) 

        # Maintain boundary conditions
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # Calculate convergence metric
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)

        # Check convergence
        iterations += 1
        if max_change < tol:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    Solve Laplace equation using Gauss-Seidel SOR iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        omega (float): Relaxation factor
        Niter (int): Maximum number of iterations
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize potential grid
    u = np.zeros((ygrid, xgrid))
    
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    convergence_history = []
    
    for iteration in range(Niter):
        u_old = u.copy()
        
        # SOR iteration
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # Skip plate regions
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # Calculate residual
                r_ij = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                
                # Apply SOR formula
                u[i, j] = (1 - omega) * u[i, j] + omega * r_ij
        
        # Maintain boundary conditions
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # Calculate convergence metric
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # Check convergence
        if max_change < tol:
            break
    
    return u, iteration + 1, convergence_history

def plot_results(x, y, u, method_name):
    """
    Plot 3D potential distribution and equipotential contours.
    
    Args:
        x (array): X coordinates
        y (array): Y coordinates
        u (array): Potential distribution
        method_name (str): Name of the method used
    """
    fig = plt.figure(figsize=(10, 5))
    
    # 3D wireframe plot
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax1.plot_wireframe(X, Y, u, alpha=0.7)
    levels = np.linspace(u.min(), u.max(), 20)
    ax1.contour(x, y, u, zdir='z', offset=u.min(), levels=levels)
    ax1.set_xlabel('X 方向')
    ax1.set_ylabel('Y 方向')
    ax1.set_zlabel('电势 (V)')
    ax1.set_title(f'三维电势分布图\n（{method_name}）')
    
    # Equipotential contour plot and Electric field streamlines combined
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 20)
    contour = ax2.contour(X, Y, u, levels=levels, colors='red', linestyles='dashed', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
    
    EY, EX = np.gradient(-u, 1)  # Electric field is negative gradient of potential
    ax2.streamplot(X, Y, EX, EY, density=1.5, color='blue', linewidth=1, arrowsize=1.5, arrowstyle='->')
    
    ax2.set_xlabel('X 方向')
    ax2.set_ylabel('Y 方向')
    ax2.set_title(f'等势线与电场线图\n（{method_name}）')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    xgrid, ygrid = 50, 50
    w, d = 20, 20  # plate width and separation
    tol = 1e-3
    # Create coordinate arrays
    x = np.linspace(0, xgrid-1, xgrid)
    y = np.linspace(0, ygrid-1, ygrid)
    
    print("求解平行板电容器的拉普拉斯方程...")
    print(f"网格大小: {xgrid} x {ygrid}")
    print(f"板宽: {w}, 板间距: {d}")
    print(f"收敛阈值: {tol}")
    
    # Solve using Jacobi method
    print("1. 雅可比迭代法:")
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol=tol)
    time_jacobi = time.time() - start_time
    print(f"   迭代次数: {iter_jacobi}")
    print(f"   用时: {time_jacobi:.3f} 秒")
    
    # Solve using SOR method
    print("2. 高斯-赛德尔 SOR 迭代法:")
    start_time = time.time()
    u_sor, iter_sor, conv_history_sor = solve_laplace_sor(xgrid, ygrid, w, d, tol=tol)
    time_sor = time.time() - start_time
    print(f"   迭代次数: {iter_sor}")
    print(f"   用时: {time_sor:.3f} 秒")
    
    # Performance comparison
    print("\n3. 性能比较:")
    print(f"   雅可比: {iter_jacobi} 次迭代, {time_jacobi:.3f} 秒")
    print(f"   SOR:    {iter_sor} 次迭代, {time_sor:.3f} 秒")
    print(f"   加速比: {iter_jacobi/iter_sor:.1f} 倍迭代速度, {time_jacobi/time_sor:.2f} 倍时间速度")
    
    # Plot results
    plot_results(x, y, u_jacobi, "雅可比迭代法")
    plot_results(x, y, u_sor, "SOR 迭代法")
    
    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(conv_history_jacobi)), conv_history_jacobi, 'r-', label='雅可比迭代法')
    plt.semilogy(range(len(conv_history_sor)), conv_history_sor, 'b-', label='SOR 迭代法')
    plt.xlabel('迭代次数')
    plt.ylabel('最大变化量')
    plt.title('收敛性比较')
    plt.grid(True)
    plt.legend()
    plt.show()
