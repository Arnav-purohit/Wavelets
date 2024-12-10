import numpy as np
import pywt  # PyWavelets for wavelet transforms
import matplotlib.pyplot as plt

# ================================================
# Parameters and Grid Setup
# ================================================
nx, ny = 64, 64  # Grid size
Lx, Ly = 1.0, 1.0  # Domain size (length in x and y directions)
dx, dy = Lx / nx, Ly / ny  # Grid spacing
Re = 100  # Reynolds number (viscosity parameter)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity and pressure fields
u = np.zeros((ny, nx))  # x-component of velocity
v = np.zeros((ny, nx))  # y-component of velocity
p = np.zeros((ny, nx))  # Pressure field

# Add a smooth initial vortex as a test case
u = np.sin(np.pi * X) * np.cos(np.pi * Y)
v =-np.cos(np.pi * X) * np.sin(np.pi * Y)

# Compute initial stable time step (CFL condition)
dt = 0.5 * min(dx, dy) / np.sqrt(2)  # Safety factor 0.5

# ================================================
# Wavelet Transform Functions
# ================================================
def apply_wavelet_transform(field, wavelet='db1', level=2):
    """
    Applies discrete wavelet transform (DWT) to the field.
    """
    return pywt.wavedec2(field, wavelet, level=level)

def reconstruct_from_wavelet(coeffs, wavelet='db1'):
    """
    Reconstructs a field from its wavelet coefficients.
    """
    return pywt.waverec2(coeffs, wavelet)

def denoise_with_wavelets(coeffs, threshold=0.1):
    """
    Applies thresholding to wavelet coefficients for denoising.
    """
    cA, *details = coeffs  # Separate approximation coefficients
    denoised_details = [
        tuple(pywt.threshold(d, threshold, mode='soft') for d in detail_level)
        for detail_level in details
    ]
    return [cA] + denoised_details

# ================================================
# Navier-Stokes Solver with Wavelets
# ================================================
def solve_navier_stokes(u, v, p, nt):
    """
    Solves the 2D Navier-Stokes equations using finite differences
    with wavelet-based filtering for spatial denoising.
    """
    global dt
    # Create a figure for subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2x3 grid for 6 subplots
    axs = axs.flatten()  # Flatten to easily index the subplots

    for step in range(nt):
        # Compute derivatives
        dudx = np.gradient(u, axis=1) / dx
        dudy = np.gradient(u, axis=0) / dy
        dvdx = np.gradient(v, axis=1) / dx
        dvdy = np.gradient(v, axis=0) / dy

        laplacian_u = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx**2 + \
                      (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dy**2
        laplacian_v = (np.roll(v, -1, axis=1) - 2 * v + np.roll(v, 1, axis=1)) / dx**2 + \
                      (np.roll(v, -1, axis=0) - 2 * v + np.roll(v, 1, axis=0)) / dy**2

        # Apply wavelet transform for spatial filtering
        coeffs_u = apply_wavelet_transform(u)
        coeffs_v = apply_wavelet_transform(v)

        # Denoising step: Threshold small coefficients
        coeffs_u = denoise_with_wavelets(coeffs_u)
        coeffs_v = denoise_with_wavelets(coeffs_v)

        # Reconstruct the filtered fields
        u_filtered = reconstruct_from_wavelet(coeffs_u)
        v_filtered = reconstruct_from_wavelet(coeffs_v)

        # Update velocity fields using the Navier-Stokes equations
        u_new = u_filtered + dt * (-u * dudx - v * dudy + (1 / Re) * laplacian_u)
        v_new = v_filtered + dt * (-u * dvdx - v * dvdy + (1 / Re) * laplacian_v)

        # Apply boundary conditions (no-slip walls)
        u_new[:, 0] = u_new[:, -1] = u_new[0, :] = u_new[-1, :] = 0
        v_new[:, 0] = v_new[:, -1] = v_new[0, :] = v_new[-1, :] = 0

        # Update fields and adjust dt dynamically
        u, v = u_new, v_new
        max_velocity = max(np.max(np.abs(u)), np.max(np.abs(v)), 1e-6)
        dt = 0.5 * min(dx, dy) / max_velocity

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            print(f"NaN detected at step {step}")
            break

        if step % 10 == 0:
            print(f"Time step {step}/{nt} completed.")
        
        # Visualize the velocity field every 20 time steps
        if step % 20 == 0:
            velocity_magnitude = np.sqrt(u_new**2 + v_new**2)
            ax = axs[step // 20]  # Choose the correct subplot
            c = ax.contourf(X, Y, velocity_magnitude, levels=50, cmap='viridis')
            ax.quiver(X, Y, u_new, v_new, scale=10, color='white')
            ax.set_title(f"Time Step {step}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid()
            fig.colorbar(c, ax=ax, label="Velocity Magnitude")
    
    # Final adjustments to the subplot layout
    plt.tight_layout()
    plt.show()

    return u, v

# ================================================
# Solve and Visualize Results
# ================================================
nt = 100  # Number of time steps
u_final, v_final = solve_navier_stokes(u, v, p, nt)

# Final visualization of the velocity field
velocity_magnitude = np.sqrt(u_final**2 + v_final**2)
plt.figure(figsize=(10, 6))
plt.contourf(X, Y, velocity_magnitude, levels=50, cmap='viridis')
plt.colorbar(label="Velocity Magnitude")
plt.quiver(X, Y, u_final, v_final, scale=10, color='white')
plt.title("Final Velocity Field After {} Time Steps".format(nt))
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()
