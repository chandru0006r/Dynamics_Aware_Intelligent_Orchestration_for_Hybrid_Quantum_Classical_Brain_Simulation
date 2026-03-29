"""
Phase 3: GPU Backend Wrapper

The GPU backend is the powerhouse executioner for batched tasks.
Unlike the CPU which uses an adaptive Scipy integrator (RK45) designed
to tiptoe carefully around stiff curves, the GPU utilizes a blunt-force
fixed-timestep 4th-order Runge-Kutta (RK4) method.

Target workload: High-parallelism, medium-stiffness tasks.
(e.g., standard spiking neurons, massive batches).
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hh_neuron import hh_derivatives

class GPUBackend:
    """Wrapper for highly parallel fixed-step ODE solving."""
    def __init__(self):
        self.solver_method = 'RK4_Fixed'

    def execute(self, state, t, dt, Iext=10.0, noise_sigma=0.0):
        """
        Executes a single time step using the GPU-optimal fixed RK4 solver.
        (Implemented via standard NumPy arrays to simulate batched CuPy)
        
        Args:
            state: Length-4 numpy array [V, m, h, n]
            t: Current time
            dt: Timestep size
            Iext: External current
            noise_sigma: Standard deviation for noise
            
        Returns:
            new_state: Updated state array after time dt
            execution_time_ms: Simulated or actual time cost
        """
        import time
        start_t = time.perf_counter()
        
        def f(tt, yy):
            return hh_derivatives(tt, yy, Iext=Iext, noise=0.0)
            
        # The classic RK4 algorithm: heavily parallelizable on GPUs
        k1 = np.array(f(t, state)) * dt
        k2 = np.array(f(t + dt/2, state + k1/2)) * dt
        k3 = np.array(f(t + dt/2, state + k2/2)) * dt
        k4 = np.array(f(t + dt, state + k3)) * dt
            
        new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Inject fast noise kernel additively to Voltage variable
        if noise_sigma > 0.0:
            new_state[0] += np.random.normal(0, noise_sigma * dt)
            
        # Hard limits to prevent explicit mathematical explosions (Biological Bounds)
        new_state[0] = np.clip(new_state[0], -150.0, 150.0)
        new_state[1:] = np.clip(new_state[1:], 0.0, 1.0)
            
        exec_ms = (time.perf_counter() - start_t) * 1000
        return new_state, exec_ms
