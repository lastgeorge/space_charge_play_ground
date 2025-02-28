import numpy as np
import matplotlib.pyplot as plt

def compute_fourier_coefficients(Lx, Ly, Lz, rho_max, epsilon, Nn, Nm, Nl):
    """
    Compute the Fourier coefficients A_{nml} for the potential expansion.
    
    For a charge distribution ρ(x) = rho_max * (x/Lx), we have
    I_n = ∫₀^{Lx} (rho_max*x/Lx)*sin(nπx/Lx) dx = rho_max * (-1)^(n+1) / (nπ).
    For y and z:
      I_m = 2Ly/(mπ) for m odd (0 for even), and
      I_l = 2Lz/(lπ) for l odd (0 for even).
    
    Then:
      A_{nml} = (32 * rho_max * (-1)^(n+1)) / (Lx * n * π^3 * m * l * λ_{nml} * epsilon)
    with λ_{nml} = (nπ/Lx)^2 + (mπ/Ly)^2 + (lπ/Lz)^2.
    Only odd m and l contribute.
    """
    pi = np.pi
    coeffs = []
    # Only consider odd m and odd l:
    m_values = [m for m in range(1, Nm+1) if m % 2 == 1]
    l_values = [l for l in range(1, Nl+1) if l % 2 == 1]
    for n in range(1, Nn+1):
        for m in m_values:
            for l in l_values:
                lam = (n*pi/Lx)**2 + (m*pi/Ly)**2 + (l*pi/Lz)**2
                A = (32.0 * rho_max * ((-1)**(n+1))) / (Lx * n * (pi**3) * m * l * lam * epsilon)
                coeffs.append((n, m, l, A))
    return coeffs

def potential_at_point(x, y, z, Lx, Ly, Lz, coeffs):
    """Evaluate the potential at (x,y,z) using the Fourier series expansion."""
    pi = np.pi
    Phi = 0.0
    for (n, m, l, A) in coeffs:
        Phi += A * np.sin(n*pi*x/Lx) * np.sin(m*pi*y/Ly) * np.sin(l*pi*z/Lz)
    return Phi

def electric_field_at_point(x, y, z, Lx, Ly, Lz, coeffs):
    """
    Compute the electric field components (Ex, Ey, Ez) at (x,y,z) 
    from the derivative of the Fourier series for the potential.
    
    E_x = -∂Φ/∂x = -Σ A_{nml} (nπ/Lx) cos(nπx/Lx) sin(mπy/Ly) sin(lπz/Lz), etc.
    """
    pi = np.pi
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    for (n, m, l, A) in coeffs:
        Ex += A * (n*pi/Lx) * np.cos(n*pi*x/Lx) * np.sin(m*pi*y/Ly) * np.sin(l*pi*z/Lz)
        Ey += A * (m*pi/Ly) * np.sin(n*pi*x/Lx) * np.cos(m*pi*y/Ly) * np.sin(l*pi*z/Lz)
        Ez += A * (l*pi/Lz) * np.sin(n*pi*x/Lx) * np.sin(m*pi*y/Ly) * np.cos(l*pi*z/Lz)
    return -Ex, -Ey, -Ez

def compute_field_on_grid(Lx, Ly, Lz, rho_max, epsilon, nx, ny, nz, Nn, Nm, Nl):
    """
    Compute the potential and electric field on a 3D grid.
    
    Returns:
      x_vals, y_vals, z_vals: 1D arrays of grid coordinates.
      Phi: 3D array of the potential.
      Ex, Ey, Ez: 3D arrays of the electric field components.
    """
    coeffs = compute_fourier_coefficients(Lx, Ly, Lz, rho_max, epsilon, Nn, Nm, Nl)
    x_vals = np.linspace(0, Lx, nx)
    y_vals = np.linspace(0, Ly, ny)
    z_vals = np.linspace(0, Lz, nz)
    
    Phi = np.zeros((nx, ny, nz))
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                Phi[i,j,k] = potential_at_point(x, y, z, Lx, Ly, Lz, coeffs)
                Ex[i,j,k], Ey[i,j,k], Ez[i,j,k] = electric_field_at_point(x, y, z, Lx, Ly, Lz, coeffs)
    
    return x_vals, y_vals, z_vals, Phi, Ex, Ey, Ez

# ----------------------------
# Validation Functions
# ----------------------------

def compute_divergence(Ex, Ey, Ez, dx, dy, dz):
    """
    Compute the divergence ∇·E on a 3D grid using finite differences.
    For interior points, use central differences.
    For boundaries, use one-sided differences.
    """
    nx, ny, nz = Ex.shape
    divE = np.zeros_like(Ex)
    
    # Central differences for interior points:
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                dEx_dx = (Ex[i+1,j,k] - Ex[i-1,j,k]) / (2*dx)
                dEy_dy = (Ey[i,j+1,k] - Ey[i,j-1,k]) / (2*dy)
                dEz_dz = (Ez[i,j,k+1] - Ez[i,j,k-1]) / (2*dz)
                divE[i,j,k] = dEx_dx + dEy_dy + dEz_dz
                
    # Boundaries: use forward/backward differences.
    # i-boundaries:
    for j in range(ny):
        for k in range(nz):
            divE[0,j,k] = (Ex[1,j,k] - Ex[0,j,k]) / dx
            divE[-1,j,k] = (Ex[-1,j,k] - Ex[-2,j,k]) / dx
    # j-boundaries:
    for i in range(nx):
        for k in range(nz):
            divE[i,0,k] += (Ey[i,1,k] - Ey[i,0,k]) / dy
            divE[i,-1,k] += (Ey[i,-1,k] - Ey[i,-2,k]) / dy
    # k-boundaries:
    for i in range(nx):
        for j in range(ny):
            divE[i,j,0] += (Ez[i,j,1] - Ez[i,j,0]) / dz
            divE[i,j,-1] += (Ez[i,j,-1] - Ez[i,j,-2]) / dz
    
    return divE

def compute_curl(Ex, Ey, Ez, dx, dy, dz):
    """
    Compute the curl of the electric field (∇×E) on a 3D grid using finite differences.
    Returns (curl_x, curl_y, curl_z).
    """
    nx, ny, nz = Ex.shape
    curl_x = np.zeros_like(Ex)
    curl_y = np.zeros_like(Ex)
    curl_z = np.zeros_like(Ex)
    
    # Use central differences for interior points:
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                dEz_dy = (Ez[i,j+1,k] - Ez[i,j-1,k]) / (2*dy)
                dEy_dz = (Ey[i,j,k+1] - Ey[i,j,k-1]) / (2*dz)
                curl_x[i,j,k] = dEz_dy - dEy_dz
                
                dEx_dz = (Ex[i,j,k+1] - Ex[i,j,k-1]) / (2*dz)
                dEz_dx = (Ez[i+1,j,k] - Ez[i-1,j,k]) / (2*dx)
                curl_y[i,j,k] = dEx_dz - dEz_dx
                
                dEy_dx = (Ey[i+1,j,k] - Ey[i-1,j,k]) / (2*dx)
                dEx_dy = (Ex[i,j+1,k] - Ex[i,j-1,k]) / (2*dy)
                curl_z[i,j,k] = dEy_dx - dEx_dy
                
    return curl_x, curl_y, curl_z

# ----------------------------
# Main Program
# ----------------------------

if __name__ == "__main__":
    # Domain parameters (in cm)
    Lx = 10.0
    Ly = 10.0
    Lz = 10.0
    
    # Charge distribution: ρ(x) = rho_max * (x/Lx)
    rho_max = 1e-5  # arbitrary units (e.g., C/cm^3)
    
    # Permittivity (for simplicity, use SI vacuum permittivity; careful with units)
    epsilon = 8.854e-12  # F/m
    
    # Grid resolution: number of grid points along each dimension
    nx = 50
    ny = 50
    nz = 50
    
    # Number of Fourier modes in each direction.
    Nn = 10   # modes in x
    Nm = 10   # modes in y (only odd modes contribute)
    Nl = 10   # modes in z (only odd modes contribute)
    
    # Compute the potential and electric field on the grid.
    x_vals, y_vals, z_vals, Phi, Ex, Ey, Ez = compute_field_on_grid(Lx, Ly, Lz, rho_max, epsilon,
                                                                     nx, ny, nz, Nn, Nm, Nl)
    
    # Compute grid spacing (assuming uniform grid)
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    dz = Lz / (nz - 1)
    
    # -----------------------
    # Validation 1: Reconstruct Charge from ∇·E
    # -----------------------
    divE = compute_divergence(Ex, Ey, Ez, dx, dy, dz)
    # Derived charge density: ρ_derived = ε * (∇⋅E)
    rho_derived = epsilon * divE
    
    # Input charge density: ρ_input(x) = rho_max * (x / Lx), independent of y and z.
    rho_input = np.zeros_like(rho_derived)
    for i, x in enumerate(x_vals):
        rho_input[i, :, :] = rho_max * (x / Lx)
    
    # Compute error between derived and input charge densities.
    error = np.abs(rho_derived - rho_input)
    max_error = np.max(error)
    mean_error = np.mean(error)
    print("Validation 1: Charge Reconstruction")
    print("Maximum absolute error in ρ:", max_error)
    print("Mean absolute error in ρ:", mean_error)
    
    # Plot a slice at mid z comparing input and derived charge densities.
    mid_z = nz // 2
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.contourf(x_vals, y_vals, rho_input[:,:,mid_z].T, 20, cmap='viridis')
    plt.title("Input ρ(x) at mid-z")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.contourf(x_vals, y_vals, rho_derived[:,:,mid_z].T, 20, cmap='viridis')
    plt.title("Derived ρ from ∇·E at mid-z")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    # -----------------------
    # Validation 2: Check Curl of E-field
    # -----------------------
    curl_x, curl_y, curl_z = compute_curl(Ex, Ey, Ez, dx, dy, dz)
    curl_magnitude = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
    max_curl = np.max(np.abs(curl_magnitude))
    mean_curl = np.mean(np.abs(curl_magnitude))
    print("\nValidation 2: Curl of E-field")
    print("Maximum |curl E|:", max_curl)
    print("Mean |curl E|:", mean_curl)
    
    # Plot histogram of the curl magnitude over the grid.
    plt.figure(figsize=(6,5))
    plt.hist(curl_magnitude.ravel(), bins=50)
    plt.title("Histogram of |curl E| over the grid")
    plt.xlabel("|curl E|")
    plt.ylabel("Counts")
    plt.show()
    
    # Additionally, plot a 2D slice of the curl magnitude at mid z.
    plt.figure(figsize=(6,5))
    plt.contourf(x_vals, y_vals, curl_magnitude[:,:,mid_z].T, 20, cmap='inferno')
    plt.title("Curl Magnitude |curl E| at mid-z")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.colorbar()
    plt.show()