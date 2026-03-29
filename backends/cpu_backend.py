"""
Phase 3: CPU Backend Wrapper

The CPU backend is the dependable sequential worker.
It natively uses the highly accurate adaptive 4th-order Runge-Kutta
method (RK45) from SciPy.

Target workload: Low-stiffness, low-parallelizability tasks
(e.g., synaptic updates or cleanly resting neurons).
"""

import sys
import os
import numpy as np
from scipy.integrate import solve_ivp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hh_neuron import hh_derivatives

class CPUBackend:
    """Wrapper for traditional CPU-based ODE solving."""
    def __init__(self):
        self.solver_method = 'RK45'
        self.rtol = 1e-6
        self.atol = 1e-8

    def execute(self, state, t, dt, Iext=10.0, noise_sigma=0.0):
        """
        Executes a single time step using the CPU RK45 solver.
        
        Args:
            state: Length-4 numpy array [V, m, h, n]
            t: Current time
            dt: Timestep size
            Iext: External current
            noise_sigma: Standard deviation for noise (not used in deterministic step but kept for api consistency)
            
        Returns:
            new_state: Updated state array after time dt
            execution_time_ms: Simulated or actual time cost
        """
        import time
        start_t = time.perf_counter()
        
        # Scipy evaluate from t to t+dt
        t_span = (t, t + dt)
        
        # We enforce strict deterministic solving for the step to maintain accuracy.
        # So we pass noise=0.0 for the bare integral.
        sol = solve_ivp(
            lambda t_eval, y: hh_derivatives(t_eval, y, Iext=Iext, noise=0.0),
            t_span, 
            state,
            method=self.solver_method,
            rtol=self.rtol, 
            atol=self.atol
        )
        
        new_state = sol.y[:, -1]
        
        # If noise was requested, we apply it additively to the voltage state
        if noise_sigma > 0.0:
            new_state[0] += np.random.normal(0, noise_sigma * dt)
            
        exec_ms = (time.perf_counter() - start_t) * 1000
        return new_state, exec_ms
