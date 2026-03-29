import sys
import os
from utils.logger import session
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends.cpu_backend import CPUBackend

def run_baseline_simulation():
    print("=" * 60)
    print("PHASE 4: Baseline CPU-Only Benchmark")
    print("=" * 60)
    
    # 1. Setup Simulation Parameters
    NUM_NEURONS = 8
    T_START = 0.0
    T_END = 100.0   # Simulate 100ms
    DT = 0.1        # 0.1ms timestep 
    STEPS = int(T_END / DT)
    
    # Storage arrays
    time_points = np.linspace(T_START, T_END, STEPS)
    voltage_traces = np.zeros((NUM_NEURONS, STEPS))
    
    # Initial state: [V=-65mV, m=0.05, h=0.6, n=0.32] for all 8 neurons
    states = np.tile([-65.0, 0.05, 0.6, 0.32], (NUM_NEURONS, 1))
    voltage_traces[:, 0] = states[:, 0]
    
    cpu = CPUBackend()
    
    cumulative_latency = 0.0
    print(f"Starting 100ms Baseline Simulation ({STEPS} steps)...")
    
    t0_wallclock = time.time()
    
    # 2. Main Time Loop
    for step in range(1, STEPS):
        t = time_points[step-1]
        
        # Inject external current stimulus
        # Neurons 0-3 spike heavily at exactly t=10ms 
        # Neurons 4-7 stay relatively quiet but get a little background noise
        for n_idx in range(NUM_NEURONS):
            if n_idx < 4 and 10.0 <= t <= 50.0:
                I_ext = 10.0 + (np.random.normal(0, 0.5) if n_idx == 0 else 0)
            else:
                I_ext = 0.5
                
            # Process sequentially (CPU pipeline)
            new_state, exec_ms = cpu.execute(
                states[n_idx].copy(), 
                t, 
                DT, 
                Iext=I_ext,
                noise_sigma=1.5 if t > 10.0 else 0.0
            ) # Baseline does not have surrogate/QPU fast noise
            
            states[n_idx] = new_state
            voltage_traces[n_idx, step] = new_state[0]
            cumulative_latency += exec_ms
            
        if step % 200 == 0:
            print(f"  Progress: t={t:.1f}ms / {T_END}ms")
            
    wallclock_total = time.time() - t0_wallclock
    
    print("\n--- RESULTS ---")
    print(f"Algorithmic Sequential Latency: {cumulative_latency:.2f} ms")
    print(f"Python Wallclock Time:          {wallclock_total*1000:.2f} ms")
    
    # 3. Plotting Verification
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot first 4 neurons
    for n_idx in range(4):
        ax.plot(time_points, voltage_traces[n_idx, :], label=f'Neuron {n_idx} (Stimulated)', alpha=0.8)
    
    # Plot one quiet neuron for reference
    ax.plot(time_points, voltage_traces[-1, :], label='Neuron 7 (Resting)', color='gray', linestyle='--')
    
    ax.set_title("Phase 4: Baseline CPU-Only Voltage Traces")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    out_dir = session.figures_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'week7_baseline_trace.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    print("============================================================")
    
    return cumulative_latency

if __name__ == "__main__":
    run_baseline_simulation()
