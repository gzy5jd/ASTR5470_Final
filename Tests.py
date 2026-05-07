# -------------------------------------
# Imports
# -------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq
from scipy.stats import linregress

import SemiLagrangian

# -------------------------------------
# Jeans Criterion Theory
# -------------------------------------
def jeans_wavenumber(G,rho0,sigma):

    return np.sqrt(4 * np.pi * G * rho0 / sigma**2)

# -------------------------------------
# Growth Rate \gamma Fitting
# -------------------------------------
def fit_growth_rate(times,amplitudes,fit_fraction=0.4):

    nfit = int(len(times) * fit_fraction)
    tfit = np.array(times[:nfit])
    afit = np.log(np.array(amplitudes[:nfit]))

    slope, intercept, *_ = linregress(tfit,afit)
    return slope

# -------------------------------------
# TEST 1 & 2 --- Jeans Instability Test / Energy Conservation
# -------------------------------------
def jeans_stability():

    # Set simulation parameters
    nx = 100
    nv = 100
    lx = 20.0
    lv = 12.0
    dt = 0.01
    G = 1.0
    sigma = 1.0

    # Predicted critical jeans wavenumber value for rho = 1
    rho0 = 1.0
    kJ = jeans_wavenumber(G,rho0,sigma)
    
    print(f"Critical Jeans wavenumber: {kJ:.4f}")

    # -----------------------------------------
    # Test a variety of modes (mode values chosen by trial and error)
    # -----------------------------------------
    modes = [1, 2, 4, 8, 12, 16]
    growth_rates = []
    k_values = []

    # -----------------------------------------
    # Loop over k-modes
    # -----------------------------------------
    for mode in modes:

        print(f"Current mode: {mode}")
        solver = SemiLagrangian.SemiLagrangian(nx=nx,nv=nv,lx=lx,lv=lv,dt=dt,G=G)

        solver.initialize_distribution(perturbation=1e-3,k_mode=mode,sigma=sigma)
        solver.run_simulation(steps=120,k_mode=mode)

        gamma = fit_growth_rate(solver.time_vals,solver.mode_vals)
        growth_rates.append(gamma)
        current_k = (2 * np.pi * mode / solver.lx)
        k_values.append(current_k)

        print(f"Growth rate: {gamma}")

        # Plot a density plot for each mode
        for i in range(len(solver.rho_vals)):
            if i % 20 == 0:
                plt.plot(solver.x,solver.rho_vals[i],label=f"t={solver.time_vals[i]:.2f}")

        plt.xlabel("x")
        plt.ylabel(r"$\rho(x)$")
        plt.title("Density")
        plt.legend(framealpha=0)
        plt.tight_layout()
        plt.savefig(f'./OUTPUT/TEST1/DensityEvolution_mode{mode}')
        plt.show()
        plt.clf()

        # -------------------------------------
        # TEST 2 --- Energy Conservation Test
        # -------------------------------------
        energy_scatter = []
        time_scatter = []
        for i in range(len(solver.energy_vals)):
            if i % 20 == 0:
                energy_scatter.append(solver.energy_vals[i])
                time_scatter.append(solver.time_vals[i])

        # Calculate percent change to add to title of plot
        plt.scatter(time_scatter, energy_scatter,color='black')
        plt.xlabel('Time')
        plt.ylabel('Energy (T+V)')
        plt.title(f'Energy Stability - % Change = {100 * (np.max(energy_scatter) - np.min(energy_scatter)) / energy_scatter[0]}')
        plt.tight_layout()
        plt.savefig(f'./OUTPUT/TEST2/Energy_mode{mode}.png')
        plt.show()
        plt.clf()

    # -----------------------------------------
    # Plot results
    # -----------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(k_values,growth_rates,color='black')
    plt.axvline(kJ,color='black',linestyle='--',label='Critical Jeans wavenumber')
    plt.xlabel("k")
    plt.ylabel(r"$\gamma$")
    plt.title("Jeans Mode Growth")
    plt.legend(framealpha=0)
    plt.tight_layout()
    plt.savefig('./OUTPUT/TEST1/Instability.png')
    plt.show()
    plt.clf()

# -------------------------------------
# TEST 3 --- Numerical Stability
# -------------------------------------
def numerical_stability():

    # Set simulation parameters
    nx = 100
    nv = 100
    lx = 20.0
    lv = 12.0
    dt = 0.01
    G = 1.0
    sigma = 1.0

    # Choose a mode which we know is stable from before, n = 16
    mode = 16
    dt_vals = [0.005, 0.01, 0.02, 0.05, 0.1]

    growth_rates = []
    percent_change = []

    # Initialize models with progressively larger timesteps
    for dt_val in dt_vals:
        
        # Initialize with given dt value
        solver = SemiLagrangian.SemiLagrangian(nx=nx,nv=nv,lx=lx,lv=lv,dt=dt_val,G=G)

        solver.initialize_distribution(perturbation=1e-3,k_mode=mode,sigma=sigma)
        solver.run_simulation(steps=100,k_mode=mode)

        # Determine stability from growth rate and energy % change

        gamma = fit_growth_rate(solver.time_vals,solver.mode_vals)
        growth_rates.append(gamma)
        current_k = (2 * np.pi * mode / solver.lx)

        print(f"Growth rate: {gamma}")

        energy_scatter = []
        time_scatter = []
        for i in range(len(solver.energy_vals)):
            if i % 20 == 0:
                energy_scatter.append(solver.energy_vals[i])
                time_scatter.append(solver.time_vals[i])

        PercentChange = 100 * (np.max(energy_scatter) - np.min(energy_scatter)) / energy_scatter[0]
        percent_change.append(PercentChange)

    # Plot energy % Change versus dt values
    plt.scatter(dt_vals, percent_change,color='black')
    plt.xlabel('dt value')
    plt.ylabel('% Change in enery')
    plt.title('Energy numerical stability')
    plt.yscale('log')
    plt.savefig('./OUTPUT/TEST3/Energy_Stability_dt.png')
    plt.show()
    plt.clf()
        
    # Plot gamma versus dt values
    plt.scatter(dt_vals, growth_rates,color='black')
    plt.xlabel('dt value')
    plt.ylabel(r"\gamma")
    plt.title('Growth rate numerical stability')
    plt.savefig('./OUTPUT/TEST3/Growthrate_Stability_dt.png')
    plt.show()
    plt.clf()

# -------------------------------------
# Main Function
# -------------------------------------
def main():

    # Run all 3 tests
    jeans_stability()
    numerical_stability()

main()
