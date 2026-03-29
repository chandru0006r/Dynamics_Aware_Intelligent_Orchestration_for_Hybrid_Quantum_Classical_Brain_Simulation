"""
Week 4: Hybrid Partitioning Algorithm (Algorithm 1)

This algorithm dynamically assigns each node in the NeuralTaskGraph
to a compute backend (CPU, GPU, or QPU).

Theory (from paper):
────────────────────
1. Calculate the Stiffness Index (ρ) and Stochasticity Factor (W(t))
   for all membrane tasks.
2. Tentative routing rule (Equation 7):
   if ρ > ρ_θ AND W > W_θ: allocate to QPU
   elif P > P_θ: allocate to GPU
   else: allocate to CPU
3. Respect resource constraints (Q_budget):
   Each QPU computation (ReBaNO proxy) takes ~log2(N) logical qubits.
   If tentative QPU allocations exceed Q_budget, keep only the highest-ρ
   tasks on QPU and downgrade the rest to the GPU.
4. Output specific bindings: mapping = {task_id: Backend}
"""

import sys
import os
from utils.logger import session
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hh_neuron import hh_derivatives
from core.metrics import DynamicsMetrics
from core.decision_engine import DEFAULT_CONFIG, get_qubits_required

class HybridPartitioner:
    """
    Implements Algorithm 1 to partition tasks in the Task Graph
    to heterogeneous backends dynamically.
    """
    
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.metrics_engine = DynamicsMetrics()

    def decide(self, rho: float, W: float, P: float) -> str:
        """
        Decision Rule (Equation 7 from paper).
        Evaluates stiffness, stochasticity, and parallelism to assign backend.
        
        Args:
            rho: Stiffness Index
            W: Stochasticity Factor
            P: Parallelizability
        Returns:
            Backend label ('QPU', 'GPU', 'CPU')
        """
        if rho > self.config.rho_theta and W > self.config.W_theta:
            return 'QPU'
        elif P > self.config.P_theta:
            return 'GPU'
        else:
            return 'CPU'

    def partition(self, task_graph, state_histories=None):
        """
        Runs Algorithm 1 on an entire NeuralTaskGraph for a timestep.
        
        Args:
            task_graph: The Graph G=(V,E) storing Task dataclass instances
            state_histories: dict matching neuron_id to list of past states [V]
        Returns:
            mapping: mapping dict {task_id: 'BACKEND'}
        """
        mapping = {}
        qpu_candidates = []
        
        # Determine number of state variables for qubit calculation
        # usually 4 for HH
        random_task = next(iter(task_graph.tasks.values()))
        n_vars = len(random_task.state) if hasattr(random_task, 'state') else 4
        qubits_per_task = get_qubits_required(n_vars)
        
        # Step 1 & 2: Calculate metrics and tentative mappings
        for task_id, task in task_graph.tasks.items():
            
            # Simple tasks like synaptic updates don't need ODE stiffness checks
            if task.task_type != 'membrane':
                # They lack stiffness/stochasticity, default W=0, ρ=1
                backend = self.decide(1.0, 0.0, task.P)
                mapping[task_id] = backend
                task.backend = backend
                continue
            
            # Complex Membrane task evaluations
            neuron_id = task.neuron_id
            
            # History is just list of voltages to test variance for W
            hist = state_histories.get(neuron_id, []) if state_histories else []
            
            # Compute dynamics
            m = self.metrics_engine.compute_all_metrics(
                deriv_func=hh_derivatives,
                state=task.state,
                state_history=hist
            )
            
            # Assign computed metrics to Task object metadata
            task.rho = m['rho']
            task.W = m['W']
            
            # Tentative Decision
            backend = self.decide(task.rho, task.W, task.P)
            
            # Stash QPU choices for budget constraint solving later
            if backend == 'QPU':
                qpu_candidates.append(task_id)
                
            mapping[task_id] = backend
            task.backend = backend
            
        # Step 3: Resource constraint handling (Budgeting)
        total_qubits_requested = len(qpu_candidates) * qubits_per_task
        
        if total_qubits_requested > self.config.Q_budget:
            # We overflowed! The QPU is full.
            # Sort the tasks by stiffness (rho), descending
            qpu_candidates.sort(key=lambda tid: task_graph.tasks[tid].rho, reverse=True)
            
            max_qpu_tasks = self.config.Q_budget // qubits_per_task
            
            # Keep the top max_qpu_tasks on the QPU
            # Downgrade the remaining tasks to the GPU
            for tid in qpu_candidates[max_qpu_tasks:]:
                mapping[tid] = 'GPU'
                task_graph.tasks[tid].backend = 'GPU'
                
        return mapping


# ===================================================================
#  SELF-TEST - Run this file directly to verify Algorithm 1
# ===================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    
    from core.task_graph import NeuralTaskGraph
    from core.decision_engine import OrchestratorConfig
    
    print("=" * 60)
    print("WEEK 4: Decision Engine & Hybrid Partitioning - Verification")
    print("=" * 60)
    
    # We create a custom config setting threshold low specifically so we 
    # can trigger the QPU assignment and test the budgeting logic
    # Default is W_theta=0.3, rho_theta=100. Let's make it artificial for testing.
    test_config = OrchestratorConfig(rho_theta=5, W_theta=0.1, Q_budget=8)
    partitioner = HybridPartitioner(config=test_config)
    
    # Build a graph with 8 Neurons
    tg = NeuralTaskGraph(num_neurons=8)
    
    # State vectors for 8 neurons:
    # 6 fast-spiking (highly stiff) and 2 resting (low stiffness)
    # HH resting state is roughly V=-65, m=0.05, h=0.6, n=0.32
    # Spike action potential state roughly V=30, m=0.99, h=0.05, n=0.5
    resting = np.array([-65.0, 0.05, 0.6, 0.32])
    spiking = np.array([30.0, 0.99, 0.05, 0.5])
    
    initial_states = [spiking.copy() for _ in range(6)] + [resting.copy() for _ in range(2)]
    
    # Simulating a noisy environment to push W > W_theta (0.1)
    np.random.seed(42)
    fake_histories = {}
    for i in range(8):
        # Add significant voltage noise history
        fake_histories[i] = [initial_states[i][0] + np.random.normal(0, 5.0) for _ in range(10)]
        
    tg.build(initial_states)
    print(f"Testing graph built with {tg.N} Neurons (16 Tasks)")
    
    # Run partitioning!
    mapping = partitioner.partition(tg, fake_histories)
    
    counts = {'CPU': 0, 'GPU': 0, 'QPU': 0}
    for tid, backend in mapping.items():
        counts[backend] += 1
        
    print(f"\nTentative Test (8 qubits max -> 4 QPU tasks max):")
    print(f"  CPU Tasks Assigns: {counts['CPU']} (Expected 8, the Synaptic Tasks)")
    print(f"  GPU Tasks Assigns: {counts['GPU']} (Expected 4, Membrane Tasks overflowed or low stiff)")
    print(f"  QPU Tasks Assigns: {counts['QPU']} (Expected 4, Membrane Tasks heavily stiff)")

    # Verify predictions
    assert counts['CPU'] == 8
    assert counts['QPU'] == 4
    assert counts['GPU'] == 4
    print("  BUDGET VERIFICATION PASSED: Saturated exactly 4 QPU tasks!")
    
    for i in range(8):
        mem_tid = i * 2
        syn_tid = mem_tid + 1
        print(f"  Neuron {i} State: {'Spiking' if i < 6 else 'Resting'}")
        print(f"     Membrane Task [T{mem_tid:02d}]: rho={tg.tasks[mem_tid].rho:7.2f}, W={tg.tasks[mem_tid].W:6.3f} => {tg.tasks[mem_tid].backend}")
        print(f"     Synaptic Task [T{syn_tid:02d}]: P=0.5 => {tg.tasks[syn_tid].backend}")
        
    # Visualize final decision breakdown!
    save_path = os.path.join(session.figures_dir, 'week4_partition_output.png')
    tg.visualize(mapping, save_path=save_path)
    print(f"\nGraph plot mapped and saved successfully to '{save_path}'")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Threshold variables correctly utilized")
    print("✅ Partitioning successfully maps graphs to backends")
    print("✅ Equation 7 logic routes stiff noisy ODEs to QPU")
    print("✅ Q_budget enforced limits overflows to GPU safely")
    print("============================================================")
