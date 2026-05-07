# ASTR5470_Final

## Project Overview / Methods

This project implements a Semi-Lagrangian solver of the Boltzmann equation. The Boltzmann equation, is a partial differential equation for the distribution function $f$ of a system. Given a gravitational potential $\Phi$ and neglibile interparticle collisions the equation takes the form

$$
\frac{\partial f}{\partial t} + \mathbf{v} \cdot \nabla_x f + \nabla_x \Phi \cdot \nabla_v f = 0
$$

As opposed to more direct methods such as solving the Poisson equation, this equation can give greater insight in to the full physics of the system, as it tracks the phase space distribution both in space and velocity, which can provide greater insight. In a very simplified model, we can use this to understand the gravitational stability of a gas/dust cloud. Through moments of the distribution, we can compute relevant physical observables, such as density, energy, etc. We can rewrite this equation to take the form $df/dt = 0$. The distribution function is then constant along characteristics, which we can use to write an implicit solver, as opposed to a traditional explicit solver, such as a finite difference method. We define a 2D grid of position and velocity values (x, v) on which $f$ is defined. 

The Semi-Lagrangian solver traces characteristics of the distribution function through the equality

$$
f(x,v,t+\Delta t) = f(x',v',t)
$$

Where $x' = x - v \Delta t$ and $v' = v + F \Delta t$. We compute the force through solving the Poisson equation $\nabla^2 \Phi = 4\pi G \rho(x)$. This is done through Fourier transforms. Since the updated $x'$ and $v'$ values fall outside of the defined grid points, values of $f$ are found through a bilinear interpolation based on current $f$ values. Values of the density, energy, wavenumber, and time are stored periodically. 

This solver is implemented through a python class in SemiLagrangian.py, and three tests of the solver are implemented in Tests.py

## Tests

We perform three tests of the code, one based on expected physical results, and two on numerical stability. 

1) We expect there to be a critical jeans wavelength (or inversely, jeans wavenumber), at which perturbing the system will cause it to become unstable, and lead to increasing density. This is roughly inspired by the jeans criterion for gravitational collapse of dust/gas clouds. We track the fourier modes of the density. We expect these fourier modes can be written in the form $\exp(\gamma t)$, where the sign of $\gamma$ determines the stability of the mode (<0 stable, >0 unstable). We track the fourier modes as a function for each perturbing k-value, and fit to find the value of $\gamma$. We expect that at some critical k-value $k_J$, the sign of $\gamma$ will switch. Upon running for a variety of modes, I do find that the growth rate of the system does decrease with increasing wavenumber, as expected, but the precise value of the critical wavenumber is not exactly in line with predictions. I suspect that this may be due to me incorrectly calculating the critical wavenumber initially, but I am not sure. Nonetheless, the expected qualitative behavior is observed, and there is a clear stable and unstable regime. 

2) We perform a numerical stability test, by tracking the total energy of the system as a function of time. The kinetic energy can be extracted as the 2nd moment (w.r.t velocity) of the distribution function $f$, and the potential energy $\Phi(x)$ is calculated already during the simulation process. At several timesteps, the total energy of the system is calculated, and at the end of the full simulation, we calculate the percent change in the total energy, as a rough estimate for the stability. The total energy of the system appears to increase over time, regardless of if the solution is stable (>kJ) or unstable (<kJ), but that for stable solutions, the percent change in the total energy of the system is much lower than that of the unstable solutions. I suspect there are ways I could better improve the energy stability, maybe by using spline interpolation rather than bilinear interpolation, but I am not 100% sure. 

3) This approach to solving the Boltzmann equation is an implicit method, and as such, I expect to be able to take larger timesteps in the simulation while maintaining numerical stability. To test this, I ran the solver using a variety of time steps, ranging from reasonably small to quite large (dt=0.1), using a perturbing k-value greater than kJ (and therefore, I expect a stable solution). For each time step, I calculated the % change in energy from the initial to final state, as well as the growth rate $\gamma$ of the system. At larger time steps, around 0.05 to 0.1, there is a clear jump in instability, but for time steps smaller than this, there is decent consistency between results, showing some numerical stability even for larger timesteps around 0.02. For explicit methods, we have often had to use much smaller timesteps than 0.02, showing some value in using an implicit method, though I am sure there are additional ways I can improve numerical stability further.

## Code Documentation

```
--- SemiLagrangian.py
--- Tests.py
|-- OUTPUT
    |-- TEST1
        --- Plots of spatial density $\rho(x)$ for different time values at different k mode values
        --- Plot of growth rate ($\gamma$) versus different k values
    |-- TEST2
        --- Plots of energy values versus time at different k mode values
    |-- TEST3
        --- Plot of stability in energy (% change) for different dt values
        --- Plot of growth rate ($\gamma$) for different dt values
```

## How to Run

Running this project is very simple. The Semi-lagrangian solver is implemented as a class in SemiLagrangian.py. All three tests are implemented in Tests.py. To recreate the project results, run Tests.py on the command line. This outputs plots to the OUTPUT folder, and in to the folders TEST1, TEST2, and TEST3. Simulation parametes, such as limits on x values, limits on v values, grid size, timestep, and number of steps are defined upon instantiating and running the solver class. These properties can be easily changed in Tests.py to customize the solver properties.

TEST1 includes plots of spatial density at different time values for different k mode values

TEST2 includes plots of energy stability versus time for different k mode values

TEST3 includes dt numerical stability plots for both total system energy and growth rate

