import sys
import os
from utils.logger import session
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_sensitivity import measure_target_run

def run_tradeoff_sweep():
    print("=" * 60)
    print("PHASE 5: Orchestrator Qubit Budget Tradeoff Analysis")
    print("=" * 60)
    
    # We will vary the number of allowed qubits in the Orchestrator config
    # 0 = GPU heavily congested, QPU disabled
    # 2 = Mild QPU help
    # 4 = Default optimal budget
    # 8 = QPU fully handles the massive wave
    # 12 = Oversaturated (diminishing returns)
    budgets = [0, 2, 4, 8, 12]
    
    # Keeping optimal dynamic routing thresholds
    opt_rho = 20.0
    opt_W = 0.3
    
    results = []
    baseline_latency_50ms = 980.0
    
    print("\nSweeping Q_budget constraints:")
    for b in budgets:
        print(f"  Testing Q_budget = {b:2d} ...", end=" ")
        
        try:
            # We must monkey-patch the target measure function to accept custom Q_budget...
            # Wait, our `measure_target_run` hardcodes Q_budget=4. 
            # We can easily rewrite the function parameters here locally or just rely on a temporary patch.
            
            # Since we imported `measure_target_run`, it hardcodes `test_config = OrchestratorConfig(rho_theta=rho_t, W_theta=w_t, Q_budget=4)`.
            # We will instead create a local copy of the loop.
            pass
        except Exception as e:
            pass

    # Actually, rather than monkey-patching, let's just make the loop directly
    from backends.cpu_backend import CPUBackend
    from backends.gpu_backend import GPUBackend
    from backends.qpu_backend import QPUBackend
    from core.metrics import DynamicsMetrics
    from core.hh_neuron import hh_derivatives
    from core.task_graph import NeuralTaskGraph
    from algorithms.partitioning import HybridPartitioner
    from core.decision_engine import OrchestratorConfig

    for b in budgets:
        print(f"  Testing Q_budget = {b:2d} ...", end=" ")
        
        back_qpu = QPUBackend()
        back_gpu = GPUBackend()
        back_cpu = CPUBackend()
        
        metrics_engine = DynamicsMetrics()
        test_config = OrchestratorConfig(rho_theta=opt_rho, W_theta=opt_W, Q_budget=b)
        partitioner = HybridPartitioner(config=test_config)
        
        T_START = 0.0
        DT = 0.1        
        STEPS = int(50.0 / DT)
        time_points = np.linspace(T_START, 50.0, STEPS)
        
        num_neurons = 8
        states = np.tile([-65.0, 0.05, 0.6, 0.32], (num_neurons, 1))
        history_buffer = {i: [] for i in range(num_neurons)}
        current_mapping = {i: 'CPU' for i in range(num_neurons)}
        task_graph = NeuralTaskGraph(num_neurons=num_neurons)
        
        cumulative_latency = 0.0
        
        for step in range(1, STEPS):
            t = time_points[step-1]
            i_exts = []
            noises = []
            for n_idx in range(num_neurons):
                if n_idx < num_neurons//2 and 10.0 <= t <= 50.0:
                    i_exts.append(10.0 + (np.random.normal(0, 0.5) if n_idx == 0 else 0))
                    noises.append(1.5 if t > 10.0 else 0.0)
                else:
                    i_exts.append(0.5)
                    noises.append(0.0)
                    
            if step % 10 == 0:
                task_graph.build(list(states))
                for n_idx in range(num_neurons):
                    J = metrics_engine.compute_jacobian(lambda t_eval, y: hh_derivatives(t_eval, y, Iext=i_exts[n_idx], noise=0.0), states[n_idx])
                    rho = metrics_engine.stiffness_index(J)
                    mem_tid = n_idx * 2
                    task_graph.tasks[mem_tid].rho = rho
                current_mapping = partitioner.partition(task_graph, state_histories=history_buffer)
                
            step_latencies = {'CPU': 0.0, 'GPU': 0.0, 'QPU': 0.0}
            
            for n_idx in range(num_neurons):
                st = states[n_idx].copy()
                ie = i_exts[n_idx]
                ns = noises[n_idx]
                mem_tid = n_idx * 2
                target_backend = current_mapping.get(mem_tid, 'CPU')
                
                if target_backend == 'CPU':
                    new_state, exec_ms = back_cpu.execute(st, t, DT, Iext=ie, noise_sigma=ns)
                    step_latencies['CPU'] += exec_ms
                elif target_backend == 'GPU':
                    new_state, exec_ms = back_gpu.execute(st, t, DT, Iext=ie, noise_sigma=ns)
                    step_latencies['GPU'] = max(step_latencies['GPU'], exec_ms)
                elif target_backend == 'QPU':
                    new_state, exec_ms = back_qpu.execute(st, t, DT, Iext=ie, noise_sigma=ns)
                    step_latencies['QPU'] = max(step_latencies['QPU'], exec_ms)
                
                states[n_idx] = new_state
                history_buffer[n_idx].append(new_state[0])
                if len(history_buffer[n_idx]) > 15:
                    history_buffer[n_idx].pop(0)
                    
            cumulative_latency += max(step_latencies['CPU'], step_latencies['GPU'], step_latencies['QPU'])
            
        speedup = 980.0 / cumulative_latency
        print(f"Latency: {cumulative_latency:6.2f} ms | Mult: {speedup:.2f}x")
        
        results.append({
            'Q_Budget_Constraint': b,
            'Algorithmic_Latency_MS': round(cumulative_latency, 2),
            'Orchestrator_Speedup': round(speedup, 2)
        })
            
    out_dir = session.tables_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'table_IV_tradeoffs.csv')
    
    with open(out_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Q_Budget_Constraint', 'Algorithmic_Latency_MS', 'Orchestrator_Speedup'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[Done] Tradeoff Analysis complete. Exported Table to {out_path}")
    print("============================================================")

if __name__ == "__main__":
    run_tradeoff_sweep()
