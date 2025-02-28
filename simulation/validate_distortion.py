import numpy as np
import matplotlib.pyplot as plt

def compute_fourier_coefficients(Lx, Ly, Lz, rho_max, epsilon, Nn, Nm, Nl):
    """
    Compute the Fourier coefficients A_{nml} for the potential expansion.
    
    Only odd m and odd l contribute because
    ∫₀^{L_y} sin(mπ y/L_y)dy = 2L_y/(mπ) for m odd (and 0 for even), and similarly for l.
    
    For n = 1,...,Nn, and for m, l odd (from 1 to Nm and Nl respectively), the coefficient is:
    
      A_{nml} = [32 * rho_max * (-1)^(n+1)] / [Lx * n * π^3 * m * l * λ_{nml} * ε],
    
    where λ_{nml} = (nπ/Lx)² + (mπ/Ly)² + (lπ/Lz)².
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
    """
    Compute the potential at (x,y,z) using the Fourier series expansion.
    """
    pi = np.pi
    Phi = 0.0
    for (n, m, l, A) in coeffs:
        Phi += A * np.sin(n*pi*x/Lx) * np.sin(m*pi*y/Ly) * np.sin(l*pi*z/Lz)
    return Phi

def electric_field_at_point(x, y, z, Lx, Ly, Lz, coeffs):
    """
    Compute the electric field components (Ex, Ey, Ez) at (x,y,z)
    from the derivative of the Fourier series for the potential.
    
    E_x = -∂Φ/∂x = -∑ A_{nml} (nπ/Lx) cos(nπx/Lx) sin(mπy/Ly) sin(lπz/Lz)
    (and similarly for E_y and E_z).
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
    Compute the potential and electric field at every grid point in the volume.
    
    nx, ny, nz: number of grid points along x, y, and z.
    Returns:
      x_vals, y_vals, z_vals: 1D arrays of grid coordinates.
      Phi: 3D array of the potential.
      Ex, Ey, Ez: 3D arrays of the electric field components.
    """
    # Compute Fourier coefficients
    coeffs = compute_fourier_coefficients(Lx, Ly, Lz, rho_max, epsilon, Nn, Nm, Nl)
    
    # Create grid coordinates (we use a uniform grid from 0 to L in each direction)
    x_vals = np.linspace(0, Lx, nx)
    y_vals = np.linspace(0, Ly, ny)
    z_vals = np.linspace(0, Lz, nz)
    
    Phi = np.zeros((nx, ny, nz))
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    
    # Evaluate potential and field at each grid point
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            for k, z in enumerate(z_vals):
                Phi[i,j,k] = potential_at_point(x, y, z, Lx, Ly, Lz, coeffs)
                Ex[i,j,k], Ey[i,j,k], Ez[i,j,k] = electric_field_at_point(x, y, z, Lx, Ly, Lz, coeffs)
    
    return x_vals, y_vals, z_vals, Phi, Ex, Ey, Ez

if __name__ == "__main__":
    # Domain dimensions (in cm)
    Lx = 10.0
    Ly = 10.0
    Lz = 10.0
    
    # Charge distribution: rho(x) = rho_max * (x / Lx)
    # Here, rho_max is given in arbitrary units (e.g., C/cm^3) for the toy model.
    rho_max = 1e-5
    
    # Permittivity (for simplicity we use vacuum permittivity; note units!)
    epsilon = 8.854e-12  # F/m; (in a fully consistent calculation, convert cm to m)
    
    # Grid resolution: number of grid points in each direction
    nx = 20
    ny = 20
    nz = 20
    
    # Number of Fourier modes in each direction.
    # For the x direction, we sum from n=1 to Nn.
    # For y and z, only odd modes contribute.
    Nn = 10   # modes in x
    Nm = 10   # modes in y (only odd values are used)
    Nl = 10   # modes in z (only odd values are used)
    
    # Compute the potential and E-field on the grid
    x_vals, y_vals, z_vals, Phi, Ex, Ey, Ez = compute_field_on_grid(Lx, Ly, Lz, rho_max, epsilon,
                                                                     nx, ny, nz, Nn, Nm, Nl)
    
    # For example, plot a slice (e.g. E_x in the x-y plane at mid z)
    mid_z_index = nz // 2
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, Ex[:,:,mid_z_index], 20, cmap='RdBu_r')
    plt.colorbar(label="E_x (arb. units)")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title("Electric Field E_x at z = {:.2f} cm".format(z_vals[mid_z_index]))
    plt.show()

    # You can similarly plot E_y, E_z, or the potential Phi.