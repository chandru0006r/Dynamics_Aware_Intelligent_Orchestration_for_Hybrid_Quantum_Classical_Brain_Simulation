"""
Test script for the Phase 3 backend wrappers.
Ensures CPU, GPU, and QPU wrappers all consistently ingest mathematical states
and map them forward in time.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.cpu_backend import CPUBackend
from backends.gpu_backend import GPUBackend
from backends.qpu_backend import QPUBackend

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: Backend Wrappers — Verification")
    print("=" * 60)

    # Standard resting state: V, m, h, n
    state_0 = np.array([-65.0, 0.05, 0.6, 0.32]) 
    t_start = 0.0
    dt_step = 0.1  # 0.1 ms step size
    I_test = 10.0  # Spiking current

    # Initialize Backends
    cpu = CPUBackend()
    gpu = GPUBackend()
    qpu = QPUBackend()

    print(f"\n[Initial State] t={t_start} ms")
    print(f"  V={state_0[0]:.2f}, m={state_0[1]:.4f}, h={state_0[2]:.4f}, n={state_0[3]:.4f}")

    print("\n--- TEST: Integration Step Verification (dt=0.1ms) ---")
    
    # Run CPU (RK45)
    state_cpu, time_cpu = cpu.execute(state_0.copy(), t_start, dt_step, Iext=I_test)
    print(f"\n[CPU Backend] -> Method: {cpu.solver_method}")
    print(f"  V={state_cpu[0]:.4f}, m={state_cpu[1]:.4f}, h={state_cpu[2]:.4f}, n={state_cpu[3]:.4f}")
    print(f"  Simulated Latency: {time_cpu:.4f} ms")

    # Run GPU (RK4 Fixed)
    state_gpu, time_gpu = gpu.execute(state_0.copy(), t_start, dt_step, Iext=I_test)
    print(f"\n[GPU Backend] -> Method: {gpu.solver_method}")
    print(f"  V={state_gpu[0]:.4f}, m={state_gpu[1]:.4f}, h={state_gpu[2]:.4f}, n={state_gpu[3]:.4f}")
    print(f"  Simulated Latency: {time_gpu:.4f} ms")

    # Run QPU (BDF Surrogate)
    state_qpu, time_qpu = qpu.execute(state_0.copy(), t_start, dt_step, Iext=I_test)
    print(f"\n[QPU Backend] -> Method: Surrogate Proxy ({qpu.solver_method})")
    print(f"  V={state_qpu[0]:.4f}, m={state_qpu[1]:.4f}, h={state_qpu[2]:.4f}, n={state_qpu[3]:.4f}")
    print(f"  Surrogate Fast Latency: {time_qpu:.4f} ms")

    # Verification Math
    # All methods calculate the same physics, so their arrays should be nearly identical.
    diff_cg = np.linalg.norm(state_cpu - state_gpu)
    diff_cq = np.linalg.norm(state_cpu - state_qpu)

    print("\n--- RESULTS ---")
    print(f"  Difference (CPU vs GPU): {diff_cg:.6f}")
    print(f"  Difference (CPU vs QPU): {diff_cq:.6f}")

    # Standard tolerance for ODE method differences
    assert diff_cg < 0.1, "GPU fixed-RK4 deviated too wildly from exact RK45!"
    assert diff_cq < 0.1, "QPU BDF proxy deviated too wildly from exact RK45!"
    
    print("\n  PASSED! All three backends successfully initialized and executed states safely.")
    print("============================================================")
