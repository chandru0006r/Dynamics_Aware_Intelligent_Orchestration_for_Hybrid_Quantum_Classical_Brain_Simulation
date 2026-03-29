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
from backends.gpu_backend import GPUBackend
from backends.qpu_backend import QPUBackend
from core.metrics import DynamicsMetrics
from core.hh_neuron import hh_derivatives
from core.task_graph import NeuralTaskGraph
from algorithms.partitioning import HybridPartitioner
from algorithms.sd_policy import SDPolicyScheduler

def run_proposed_simulation():
    print("=" * 60)
    print("PHASE 4: Proposed Hybrid Q-Orchestrator Benchmark")
    print("=" * 60)
    
    # 1. Setup Simulation Parameters
    NUM_NEURONS = 8
    T_START = 0.0
    T_END = 100.0   
    DT = 0.1        
    STEPS = int(T_END / DT)
    
    time_points = np.linspace(T_START, T_END, STEPS)
    voltage_traces = np.zeros((NUM_NEURONS, STEPS))
    states = np.tile([-65.0, 0.05, 0.6, 0.32], (NUM_NEURONS, 1))
    voltage_traces[:, 0] = states[:, 0]
    
    # Architectures
    back_qpu = QPUBackend()
    back_gpu = GPUBackend()
    back_cpu = CPUBackend()
    
    metrics_engine = DynamicsMetrics()
    from core.decision_engine import OrchestratorConfig
    test_config = OrchestratorConfig(rho_theta=20.0, W_theta=0.3, Q_budget=4)
    partitioner = HybridPartitioner(config=test_config)
    task_graph = NeuralTaskGraph()
    scheduler_cpu = SDPolicyScheduler(sd_theta=10.0)
    scheduler_gpu = SDPolicyScheduler(sd_theta=1.5)
    scheduler_qpu = SDPolicyScheduler(sd_theta=5.0)

    print(f"Starting 100ms Proposed Orchestrator Simulation ({STEPS} steps)...")
    
    cumulative_latency = 0.0
    t0_wallclock = time.time()
    
    # Maintain sliding window of recent voltages for W(t) metric calculation
    history_buffer = {i: [] for i in range(NUM_NEURONS)}
    
    # Block Scheduler - Real systems don't re-route every 0.1ms; they route every 1.0ms
    current_mapping = {i: 'CPU' for i in range(NUM_NEURONS)}
    
    # 2. Main Time Loop
    for step in range(1, STEPS):
        t = time_points[step-1]
        
        # Inject realistic stimulus (exactly matching baseline)
        i_exts = []
        noises = []
        for n_idx in range(NUM_NEURONS):
            if n_idx < 4 and 10.0 <= t <= 50.0:
                i_exts.append(10.0 + (np.random.normal(0, 0.5) if n_idx == 0 else 0))
                noises.append(1.5 if t > 10.0 else 0.0)
            else:
                i_exts.append(0.5)
                noises.append(0.0)
        
        # --- ORCHESTRATOR ROUTING LAYER (Triggered every 1.0ms) ---
        if step % 10 == 0:
            task_graph = NeuralTaskGraph(num_neurons=NUM_NEURONS)
            task_graph.build(list(states))
            
            for n_idx in range(NUM_NEURONS):
                J = metrics_engine.compute_jacobian(lambda t_eval, y: hh_derivatives(t_eval, y, Iext=i_exts[n_idx], noise=0.0), states[n_idx])
                rho = metrics_engine.stiffness_index(J)
                W = min(1.0, noises[n_idx] * 0.5)
                mem_tid = n_idx * 2
                task_graph.tasks[mem_tid].rho = rho
                
            current_mapping = partitioner.partition(task_graph, state_histories=history_buffer)
            
            # Note: We technically apply SD-Policy here to allocate fractions, 
            # but for algorithmic runtime comparison, we only need the exact partitions.
        
        # --- HARDWARE EXECUTION LAYER ---
        step_latencies = {'CPU': 0.0, 'GPU': 0.0, 'QPU': 0.0}
        
        for n_idx in range(NUM_NEURONS):
            st = states[n_idx].copy()
            ie = i_exts[n_idx]
            ns = noises[n_idx]
            mem_tid = n_idx * 2
            target_backend = current_mapping.get(mem_tid, 'CPU')
            
            if target_backend == 'CPU':
                new_state, exec_ms = back_cpu.execute(st, t, DT, Iext=ie, noise_sigma=ns)
                step_latencies['CPU'] += exec_ms  # Sequential
            elif target_backend == 'GPU':
                new_state, exec_ms = back_gpu.execute(st, t, DT, Iext=ie, noise_sigma=ns)
                step_latencies['GPU'] = max(step_latencies['GPU'], exec_ms) # Parallel batch
            elif target_backend == 'QPU':
                new_state, exec_ms = back_qpu.execute(st, t, DT, Iext=ie, noise_sigma=ns)
                step_latencies['QPU'] = max(step_latencies['QPU'], exec_ms) # Parallel VQE/Proxy
            
            states[n_idx] = new_state
            voltage_traces[n_idx, step] = new_state[0]
            
            # Update history buffer tracking last 15 states
            history_buffer[n_idx].append(new_state[0])
            if len(history_buffer[n_idx]) > 15:
                history_buffer[n_idx].pop(0)
            
        # Overall algorithmic step latency is the slowest parallel bottleneck
        bottleneck = max(step_latencies['CPU'], step_latencies['GPU'], step_latencies['QPU'])
        cumulative_latency += bottleneck
        
        if step % 200 == 0:
            print(f"  Progress: t={t:.1f}ms / {T_END}ms. Current Routing Map: {current_mapping}")
            
    wallclock_total = time.time() - t0_wallclock
    
    print("\n--- RESULTS ---")
    print(f"Algorithmic Parallel Latency: {cumulative_latency:.2f} ms")
    print(f"Python Wallclock Time:        {wallclock_total*1000:.2f} ms")
    
    # 3. Plotting Verification
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for n_idx in range(4):
        ax.plot(time_points, voltage_traces[n_idx, :], label=f'Neuron {n_idx} (Stiff)', alpha=0.8)
    ax.plot(time_points, voltage_traces[-1, :], label='Neuron 7 (Synaptic Rest)', color='gray', linestyle='--')
    
    ax.set_title("Phase 4: Proposed Hybrid-Quantum Voltage Traces")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Voltage (mV)")
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    out_dir = session.figures_dir
    out_path = os.path.join(out_dir, 'week7_proposed_trace.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    print("============================================================")
    
    return cumulative_latency

if __name__ == "__main__":
    run_proposed_simulation()
