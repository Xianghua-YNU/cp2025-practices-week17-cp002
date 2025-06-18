import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import laplace
import os
matplotlib.use('Agg')

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using Successive Over-Relaxation (SOR) method
    for finite thickness parallel plate capacitor.
    """
    # Initialize potential grid
    U = np.zeros((ny, nx))

    # Create conductor mask
    conductor_mask = np.zeros((ny, nx), dtype=bool)

    # Define conductor regions
    # Upper plate: +100V
    conductor_left = nx // 4
    conductor_right = nx // 4 * 3
    y_upper_start = ny // 2 + plate_separation // 2
    y_upper_end = y_upper_start + plate_thickness
    conductor_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    U[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0

    # Lower plate: -100V
    y_lower_end = ny // 2 - plate_separation // 2
    y_lower_start = y_lower_end - plate_thickness
    conductor_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    U[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0

    # Boundary conditions: grounded sides
    U[:, 0] = 0.0
    U[:, -1] = 0.0
    U[0, :] = 0.0
    U[-1, :] = 0.0

    # SOR iteration
    for iteration in range(max_iter):
        U_old = U.copy()
        max_error = 0.0

        # Update interior points (excluding conductors and boundaries)
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                if not conductor_mask[i, j]:  # Skip conductor points
                    # SOR update formula
                    U_new = 0.25 * (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1])
                    U[i, j] = (1 - omega) * U[i, j] + omega * U_new

                    # Track maximum error
                    error = abs(U[i, j] - U_old[i, j])
                    max_error = max(max_error, error)

        # Check convergence
        if max_error < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
    else:
        print(f"Warning: Maximum iterations ({max_iter}) reached")

    return U


def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation: rho = -1/(4*pi) * nabla^2(U)
    """
    # Calculate Laplacian using scipy.ndimage.laplace
    laplacian_U = laplace(potential_grid, mode='nearest') / (dx ** 2)  # Assuming dx=dy
    # Charge density from Poisson equation: rho = -1/(4*pi) * nabla^2(U)
    rho = -laplacian_U / (4 * np.pi)
    return rho


def plot_results(potential, charge_density, x_coords, y_coords, save_path=None):
    """
    Create comprehensive visualization of results and optionally save to file
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(15, 6))

    # Subplot 1: 3D Visualization of Potential
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, potential, rstride=3, cstride=3, color='r')
    levels = np.linspace(potential.min(), potential.max(), 20)
    ax1.contour(X, Y, potential, zdir='z', offset=potential.min(), levels=levels)
    ax1.set_title('3D Visualization of Potential')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Potential')

    # Subplot 2: 3D Charge Density Distribution
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, charge_density, cmap='RdBu_r', edgecolor='none')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Charge Density')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Charge Density')
    ax2.set_title('3D Charge Density Distribution')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存到: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Simulation parameters
    nx, ny = 120, 100  # Grid dimensions
    plate_thickness = 10  # Conductor thickness in grid points
    plate_separation = 40  # Distance between plates
    omega = 1.9  # SOR relaxation factor

    # Physical dimensions
    Lx, Ly = 1.0, 1.0  # Domain size
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    # Create coordinate arrays
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)

    print("Solving finite thickness parallel plate capacitor...")
    print(f"Grid size: {nx} x {ny}")
    print(f"Plate thickness: {plate_thickness} grid points")
    print(f"Plate separation: {plate_separation} grid points")
    print(f"SOR relaxation factor: {omega}")

    # Solve Laplace equation
    start_time = time.time()
    potential = solve_laplace_sor(
        nx, ny, plate_thickness, plate_separation, omega
    )
    solve_time = time.time() - start_time

    print(f"Solution completed in {solve_time:.2f} seconds")

    # Calculate charge density
    charge_density = calculate_charge_density(potential, dx, dy)

    # 保存路径
    save_dir = r"C:\Users\31025\OneDrive\桌面\t"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "capacitor_result.png")

    # Visualize results and save image
    plot_results(potential, charge_density, x_coords, y_coords, save_path=save_path)

    # Print some statistics
    print(f"\nPotential statistics:")
    print(f"  Minimum potential: {np.min(potential):.2f} V")
    print(f"  Maximum potential: {np.max(potential):.2f} V")
    print(f"  Potential range: {np.max(potential) - np.min(potential):.2f} V")

    print(f"\nCharge density statistics:")
    print(f"  Maximum charge density: {np.max(np.abs(charge_density)):.6f}")
    print(f"  Total positive charge: {np.sum(charge_density[charge_density > 0]) * dx * dy:.6f}")
    print(f"  Total negative charge: {np.sum(charge_density[charge_density < 0]) * dx * dy:.6f}")
    print(f"  Total charge: {np.sum(charge_density) * dx * dy:.6f}")
