# -------------------------------------
# Imports
# -------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq
from scipy.stats import linregress

# -------------------------------------
# Use of AI Disclosure
# -------------------------------------
# -- I used AI for a few parts of this project, described below: --
# 1. When first starting to write code, I asked for formatting suggestions, and the idea of using a class to implement the solver was presented
# I chose to follow this suggestion, based on having previously used other solver classes in modules like scipy

# 2. I spent a while trying to successfully write a stable and working bilinear interpolation function, without success, I eventually used an implementation
# suggested by AI, which I tested to ensure it worked as I suspected it did

# 3. I ran in to a few numerical instabilities when taking derivatvies and interpolating values, I added some numerical stability steps
# namely a smoothing filter post fft, and clipping velocity values following suggestions from AI

# -------------------------------------
# Semi Lagrangian Class
# -------------------------------------
class SemiLagrangian:
    
    # -------------------------------------
    # Global Properties/init
    # -------------------------------------
    def __init__(self, nx=128, nv=128, lx=20, lv=10, dt=0.01, G=1): # Work in natural units

        # Global simulator properties
        self.nx=nx
        self.nv=nv
        self.lx=lx
        self.lv=lv
        self.dt=dt
        self.G=G

        # Define grids
        self.x = np.linspace(0, lx, nx, endpoint=False)
        # self.x = np.linspace(-lx/2, lx/2, nx)
        self.v = np.linspace(-lv/2, lv/2, nv)

        # Define grid differences
        self.dx = self.x[1] - self.x[0]
        self.dv = self.v[1] - self.v[0]

        # Meshgrid definition
        self.X, self.V = np.meshgrid(self.x, self.v, indexing='ij')

        # Initialize f as empty grid
        self.f = np.zeros((nx,nv))

        # For saving properties while running
        self.rho_vals = []
        self.time_vals = []
        self.kmode_vals = []
        self.energy_vals = []

    # -------------------------------------
    # Density
    # -------------------------------------
    def density(self):
        return np.sum(self.f, axis=1) * self.dv
    
    # -------------------------------------
    # Kinetic Energy
    # -------------------------------------
    def kinetic_energy(self):
        return 0.5 * np.sum(self.f * self.V**2) * self.dx * self.dv

    def potential_energy(self, rho):
        # Transform and fourier transform rho
        rho = rho - np.mean(rho)
        rho_k = fft(rho)
        k = 2 * np.pi * fftfreq(self.nx, d=self.dx) # fftfreq returns frequencies used for fft of rho, convert in to wavenumbers k
        k[0] = 1e-6 # Avoid divide by 0

        # Spectral smoothing filter - this was an AI suggestion, smooths rho_k to avoid numerical issues in the derivative
        spectral_filter = np.exp(-(k**2) * 0.1)

        rho_k *= spectral_filter

        # Compute phi_k and inverse fourier transform to get back to phi(x)
        phi_k = -4 * np.pi * self.G * rho_k / (k**2)
        phi_k[0] = 0

        phi = np.real(ifft(phi_k)) # Keep only real part

        return np.sum(phi)
    
    # -------------------------------------
    # Initial Conditions
    # -------------------------------------
    def initialize_distribution(self, perturbation=1e-3, k_mode=1, sigma=1):

        # Initialize speeds as a gaussian / maxwell-boltzmann
        f0 = 10 * np.exp(-self.V**2 / (2 * sigma**2)) # Initialize with meshgrid V

        # Add an oscillatory perturbation of wavenumber k = k_mode * (1/N)
        perturb = 1 + perturbation*np.cos(2 * np.pi * k_mode * self.X / self.lx)

        self.f = f0 * perturb # perturb gaussian

        # Normalize by mean density to get approx same order of magnitude in x and v
        # rho = np.sum(self.f, axis=1) * self.dv
        rho = self.density()
        self.f /= np.mean(rho)
    
    # -------------------------------------
    # Poisson Solver/Force
    # -------------------------------------
    def force(self, rho):
        #
        # Solve Poisson equation via Fourier Transform
        #

        # Transform and fourier transform rho
        rho = rho - np.mean(rho)
        rho_k = fft(rho)
        k = 2 * np.pi * fftfreq(self.nx, d=self.dx) # fftfreq returns frequencies used for fft of rho, convert in to wavenumbers k
        k[0] = 1e-6 # Avoid divide by 0

        # Spectral smoothing filter - this was an AI suggestion, smooths rho_k to avoid numerical issues in the derivative
        spectral_filter = np.exp(-(k**2) * 0.1)

        rho_k *= spectral_filter

        # Compute phi_k and inverse fourier transform to get back to phi(x)
        phi_k = -4 * np.pi * self.G * rho_k / (k**2)
        phi_k[0] = 0

        phi = np.real(ifft(phi_k)) # Keep only real part

        #
        # Compute force from Poisson soln. 
        #
        force = -np.gradient(phi, self.dx)

        return force
    
    # -------------------------------------
    # Interpolation - After failing for a while to write a bilinear interpretation function, I used AI to debug/write
    # -------------------------------------
    def interpolate(self, f, xq, vq):
        # Periodic spatial boundaries
        xq = np.mod(xq, self.lx)

        # Velocity clipping
        vq = np.clip(vq,self.v[0],self.v[-1])

        # Cell indices
        ix = (xq / self.dx).astype(int)
        iv = ((vq - self.v[0]) / self.dv).astype(int)

        # Keep indices in bounds
        ix = np.clip(ix, 0, self.nx - 2)
        iv = np.clip(iv, 0, self.nv - 2)

        # Fractional coordinates
        dx = ((xq - self.x[ix]) / self.dx)

        dv = ((vq - self.v[iv]) / self.dv)

        # Clamp weights
        dx = np.clip(dx, 0, 1)
        dv = np.clip(dv, 0, 1)

        # Bilinear interpolation
        f_interp = (
            (1 - dx) * (1 - dv) * f[ix, iv]
            + dx * (1 - dv) * f[ix + 1, iv]
            + (1 - dx) * dv * f[ix, iv + 1]
            + dx * dv * f[ix + 1, iv + 1]
        )

        return f_interp
    
    # -------------------------------------
    # Get Peak Fourier Mode
    # -------------------------------------
    def mode_amplitude(self,rho,k_mode):
        
        rho_k = fft(rho)
        return np.abs(rho_k[k_mode]) # Compute fourier transform of density at k mode

    # -------------------------------------
    # Timestep
    # -------------------------------------
    def step(self):
        rho = self.density()
        force = self.force(rho)
        f_new = np.zeros_like(self.f) # Initialze empty to store new distribution

        # Loop over space/velocity grid
        for i in range(self.nx):
            for j in range(self.nv):

                # Get x and v value at indices i,j
                x = self.x[i]
                v = self.v[j]

                # Compute updated x and v coordinates along characteristic
                v_new = v - force[i] * self.dt

                x_new = x - v * self.dt # compute new x with new v

                # Prevent introduction of NaNs
                if not np.isfinite(v_new):
                    v_prev = 0.0

                if not np.isfinite(x_new):
                    x_prev = 0.0

                # Using new x', v' along characteristic, do bilinear interpolation to find new f value
                f_new[i,j] = self.interpolate(self.f, x_new, v_new)

        # Remove any NaNs
        self.f = np.nan_to_num(f_new)
    
    # -------------------------------------
    # Full Simulation / Solver Run
    # -------------------------------------
    def run_simulation(self, steps=100, k_mode=1):

        self.rho_vals = []
        self.time_vals = []
        self.mode_vals = []

        for n in range(steps):

            self.step()

            rho = self.density()

            # Append observables
            self.rho_vals.append(rho.copy())
            self.time_vals.append(n*self.dt)
            self.mode_vals.append(self.mode_amplitude(rho, k_mode))
            self.energy_vals.append(self.kinetic_energy() + self.potential_energy(rho))

            if n % 25 == 0:
                print(f"Step: {n}, Max density: {np.max(rho)}")



