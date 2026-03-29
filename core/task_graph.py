"""
Week 3: Neural Task Graph — G = (V, E)

This module builds the DIRECTED TASK GRAPH that represents the entire
neural simulation workload. This is the data structure the orchestrator
uses to decide what runs where.

Theory (from the paper, Section III.B):
───────────────────────────────────────
Each neuron generates multiple subtasks:
  - "membrane" task: integrate the 4 HH ODEs (V, m, h, n) — expensive
  - "synaptic" task: update synaptic conductances — depends on membrane

These become NODES in the graph. EDGES represent data dependencies:
  - Within a neuron: membrane → synapse (must solve V before updating synapses)
  - Across neurons:  neuron_i membrane → neuron_j synapse (all-to-all coupling)

Key equations:
  Eq 1: A[i][j] = 1 if edge vi → vj exists (adjacency matrix)
  Eq 2: D[i][j] = bytes(vi→vj) × A[i][j]   (data weight matrix)

For 8 neurons:
  - 16 nodes (8 membrane + 8 synaptic)
  - 8 intra-neuron edges (membrane → synapse within each neuron, 32 bytes)
  - 56 cross-neuron edges (all-to-all connectivity, 8 bytes each)
  - Total: 64 edges

Why this matters:
  The task graph tells the orchestrator WHAT depends on WHAT.
  You can't compute neuron j's synapse until neuron i's membrane is done 
  (if they're connected). The orchestrator uses this to schedule tasks
  efficiently across CPU/GPU/QPU without breaking dependencies.
"""

import numpy as np
import networkx as nx
import os
from utils.logger import session
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Task:
    """
    A single computational subtask in the neural simulation.
    
    Each neuron produces 2 tasks:
      - membrane: integrate HH ODEs (dV/dt, dm/dt, dh/dt, dn/dt)
      - synaptic: update synaptic weights based on membrane output
    
    Attributes:
        id:        Unique task identifier
        neuron_id: Which neuron this task belongs to
        task_type: 'membrane' or 'synaptic'
        state:     Current state vector [V, m, h, n]
        rho:       Stiffness index (computed by metrics module)
        W:         Stochasticity factor (computed by metrics module)
        P:         Parallelizability index (0 to 1)
        backend:   Assigned backend: 'CPU', 'GPU', or 'QPU'
    """
    id:        int
    neuron_id: int
    task_type: str             # 'membrane' or 'synaptic'
    state:     np.ndarray
    rho:       float = 0.0
    W:         float = 0.0
    P:         float = 0.5
    backend:   str   = 'CPU'


class NeuralTaskGraph:
    """
    Builds and manages the directed task graph G = (V, E) for a neural
    simulation workload.
    
    For N neurons, creates:
      - 2N nodes: N membrane tasks + N synaptic tasks
      - N intra-neuron edges: membrane_i → synapse_i  (32 bytes)
      - N*(N-1) cross-neuron edges: membrane_i → synapse_j  (8 bytes)
    
    Usage:
        tg = NeuralTaskGraph(num_neurons=8)
        tg.build(initial_states)
        A = tg.get_adjacency_matrix()    # Equation 1
        D = tg.get_weight_matrix()       # Equation 2
        tg.visualize(mapping)            # Color-coded graph
    """
    
    def __init__(self, num_neurons=8):
        self.N = num_neurons
        self.graph = nx.DiGraph()
        self.tasks: Dict[int, Task] = {}
    
    def build(self, initial_states):
        """
        Build the full task graph for N neurons.
        
        For each neuron, creates 2 tasks:
          - Task A (membrane): HH ODE integration, P=0.8 (highly parallelizable)
          - Task B (synaptic): synapse update, P=0.5 (moderately parallelizable)
        
        Edges:
          - membrane_i → synapse_i : intra-neuron dependency (32 bytes)
          - membrane_i → synapse_j : cross-neuron coupling (8 bytes per pair)
        
        Parameters:
            initial_states: list of N state vectors, each [V, m, h, n]
            
        Returns:
            self (for method chaining)
        """
        task_id = 0
        
        for n_id in range(self.N):
            # Task A: membrane integration (the main expensive task)
            t_mem = Task(
                id=task_id,
                neuron_id=n_id,
                task_type='membrane',
                state=initial_states[n_id].copy(),
                P=0.8    # highly parallelizable across neurons
            )
            self.tasks[task_id] = t_mem
            self.graph.add_node(task_id, **{
                'label':  f'N{n_id}\nmembrane',
                'neuron': n_id,
                'type':   'membrane'
            })
            mem_id = task_id
            task_id += 1
            
            # Task B: synaptic update (depends on membrane result)
            t_syn = Task(
                id=task_id,
                neuron_id=n_id,
                task_type='synaptic',
                state=initial_states[n_id].copy(),
                P=0.5    # moderate parallelism
            )
            self.tasks[task_id] = t_syn
            self.graph.add_node(task_id, **{
                'label':  f'N{n_id}\nsynapse',
                'neuron': n_id,
                'type':   'synaptic'
            })
            syn_id = task_id
            
            # Intra-neuron edge: membrane → synapse (within same neuron)
            # 32 bytes: full state vector [V, m, h, n] = 4 × 8-byte float
            self.graph.add_edge(mem_id, syn_id, bytes=32)
            task_id += 1
        
        # Cross-neuron edges: all-to-all connectivity
        # Each neuron's membrane output feeds ALL other neurons' synapses
        # 8 bytes per edge: just the voltage V (1 × 8-byte float)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    mem_task_id = i * 2       # membrane task of neuron i
                    syn_task_id = j * 2 + 1   # synapse task of neuron j
                    self.graph.add_edge(mem_task_id, syn_task_id, bytes=8)
        
        return self
    
    def get_adjacency_matrix(self):
        """
        Equation 1: A[i][j] = 1 if vi -> vj, else 0
        
        Binary matrix showing which tasks depend on which.
        For 8 neurons (16 tasks): A is 16x16
        
        Returns:
            A: n x n binary adjacency matrix (numpy array)
        """
        n = len(self.tasks)
        A = np.zeros((n, n), dtype=int)
        for (i, j) in self.graph.edges():
            A[i][j] = 1
        return A
    
    def get_weight_matrix(self):
        """
        Equation 2: D[i][j] = bytes(vi -> vj) * A[i][j]
        
        Weighted matrix showing data transfer cost (in bytes) between tasks.
        Intra-neuron edges = 32 bytes (full state)
        Cross-neuron edges = 8 bytes (just voltage)
        
        Returns:
            D: n x n weight matrix (numpy array)
        """
        n = len(self.tasks)
        D = np.zeros((n, n))
        for (i, j, data) in self.graph.edges(data=True):
            D[i][j] = data.get('bytes', 0)
        return D
    
    def cross_backend_cost(self, mapping):
        """
        Compute cross-backend data transfer cost.
        
        This is the SECONDARY OBJECTIVE of the orchestrator (Eq. 5):
        minimize the total bytes transferred between different backends.
        
        If task i runs on GPU and task j runs on QPU, the data must be
        transferred across backend boundaries — this is expensive!
        
        Parameters:
            mapping: dict {task_id: backend_name}
            
        Returns:
            total: total cross-backend transfer cost in bytes
        """
        total = 0.0
        D = self.get_weight_matrix()
        for (i, j) in self.graph.edges():
            if mapping.get(i) != mapping.get(j):
                total += D[i][j]
        return total
    
    def get_task_summary(self):
        """Get a summary of all tasks and their properties."""
        summary = []
        for tid, task in sorted(self.tasks.items()):
            summary.append({
                'id': task.id,
                'neuron': task.neuron_id,
                'type': task.task_type,
                'P': task.P,
                'rho': task.rho,
                'W': task.W,
                'backend': task.backend
            })
        return summary
    
    def visualize(self, mapping=None, save_path=None):
        """
        Plot the task graph with color-coded backends.
        
        Colors:
          - lightgray:       CPU (orchestration, simple tasks)
          - cornflowerblue:  GPU (parallel ODE integration)
          - tomato:          QPU (stiff + stochastic subtasks)
        
        Parameters:
            mapping:   dict {task_id: backend_name}, or None for unassigned
            save_path: file path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 8))
        
        color_map = {
            'CPU': 'lightgray',
            'GPU': 'cornflowerblue',
            'QPU': 'tomato',
            None:  'white'
        }
        
        pos = nx.spring_layout(self.graph, seed=42, k=2)
        colors = [
            color_map.get(mapping.get(n) if mapping else None, 'lightgray')
            for n in self.graph.nodes()
        ]
        
        labels = {
            n: self.graph.nodes[n].get('label', str(n))
            for n in self.graph.nodes()
        }
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos, 
            edge_color='gray', width=0.8, 
            arrows=True, arrowsize=12, alpha=0.6
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_color=colors, node_size=1200,
            edgecolors='black', linewidths=1.0
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos, labels=labels,
            font_size=7, font_weight='bold'
        )
        
        # Legend
        patches = [
            plt.Rectangle((0, 0), 1, 1, color='lightgray',       label='CPU'),
            plt.Rectangle((0, 0), 1, 1, color='cornflowerblue',  label='GPU'),
            plt.Rectangle((0, 0), 1, 1, color='tomato',          label='QPU')
        ]
        plt.legend(handles=patches, loc='upper left', fontsize=10)
        plt.title('Neural Task Graph G=(V,E) - 8 Neurons',
                  fontsize=13, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Task graph saved: {save_path}")
        
        plt.show()


# ===================================================================
#  SELF-TEST - Run this file directly to verify everything works
# ===================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("WEEK 3: Neural Task Graph G=(V,E) - Verification")
    print("=" * 60)
    
    # --- TEST 1: Build task graph for 8 neurons ---
    print("\n--- TEST 1: Build Task Graph for 8 Neurons ---")
    initial_states = [np.array([-65.0, 0.05, 0.6, 0.32]) for _ in range(8)]
    
    tg = NeuralTaskGraph(num_neurons=8)
    tg.build(initial_states)
    
    print(f"  Neurons:           {tg.N}")
    print(f"  Tasks created:     {len(tg.tasks)}")
    print(f"  Edges created:     {tg.graph.number_of_edges()}")
    print(f"  Expected tasks:    16 (8 membrane + 8 synaptic)")
    print(f"  Expected edges:    64 (8 intra + 56 cross-neuron)")
    
    # Verify
    assert len(tg.tasks) == 16, f"Expected 16 tasks, got {len(tg.tasks)}"
    assert tg.graph.number_of_edges() == 64, f"Expected 64 edges, got {tg.graph.number_of_edges()}"
    print("  PASSED!")
    
    # --- TEST 2: Task types ---
    print("\n--- TEST 2: Task Types ---")
    membrane_tasks = [t for t in tg.tasks.values() if t.task_type == 'membrane']
    synaptic_tasks = [t for t in tg.tasks.values() if t.task_type == 'synaptic']
    print(f"  Membrane tasks:    {len(membrane_tasks)} (P=0.8)")
    print(f"  Synaptic tasks:    {len(synaptic_tasks)} (P=0.5)")
    print(f"  Task IDs (membrane): {[t.id for t in membrane_tasks]}")
    print(f"  Task IDs (synaptic): {[t.id for t in synaptic_tasks]}")
    assert len(membrane_tasks) == 8
    assert len(synaptic_tasks) == 8
    print("  PASSED!")
    
    # --- TEST 3: Adjacency matrix A (Equation 1) ---
    print("\n--- TEST 3: Adjacency Matrix A (Equation 1) ---")
    A = tg.get_adjacency_matrix()
    print(f"  Shape:             {A.shape}")
    print(f"  Non-zero entries:  {np.sum(A)} (= total edges)")
    print(f"  Sparsity:          {1 - np.sum(A)/(A.shape[0]*A.shape[1]):.1%}")
    print(f"\n  First 6x6 block of A:")
    print(f"  (rows=source, cols=destination)")
    header = "       " + "  ".join(f"T{j:2d}" for j in range(6))
    print(f"  {header}")
    for i in range(6):
        row = "  ".join(f" {A[i,j]:2d}" for j in range(6))
        task_type = "mem" if i % 2 == 0 else "syn"
        print(f"  T{i:2d}({task_type}): {row}")
    
    assert A.shape == (16, 16)
    assert np.sum(A) == 64
    print("  PASSED!")
    
    # --- TEST 4: Weight matrix D (Equation 2) ---
    print("\n--- TEST 4: Weight Matrix D (Equation 2) ---")
    D = tg.get_weight_matrix()
    print(f"  Shape:             {D.shape}")
    print(f"  Total bytes:       {np.sum(D):.0f}")
    
    # Check intra-neuron edges (32 bytes each)
    intra_bytes = sum(D[i*2, i*2+1] for i in range(8))
    # Check cross-neuron edges (8 bytes each)
    cross_bytes = np.sum(D) - intra_bytes
    
    print(f"  Intra-neuron:      {intra_bytes:.0f} bytes (8 edges x 32 bytes)")
    print(f"  Cross-neuron:      {cross_bytes:.0f} bytes (56 edges x 8 bytes)")
    print(f"  Expected intra:    256 bytes (8 x 32)")
    print(f"  Expected cross:    448 bytes (56 x 8)")
    
    assert intra_bytes == 256, f"Expected 256 intra bytes, got {intra_bytes}"
    assert cross_bytes == 448, f"Expected 448 cross bytes, got {cross_bytes}"
    print("  PASSED!")
    
    # --- TEST 5: Cross-backend cost ---
    print("\n--- TEST 5: Cross-Backend Data Transfer Cost ---")
    
    # Scenario A: All on same backend (cost = 0)
    same_mapping = {i: 'GPU' for i in range(16)}
    cost_same = tg.cross_backend_cost(same_mapping)
    print(f"  All on GPU:        {cost_same:.0f} bytes (expect 0)")
    
    # Scenario B: Half on GPU, half on QPU
    split_mapping = {i: ('GPU' if i < 8 else 'QPU') for i in range(16)}
    cost_split = tg.cross_backend_cost(split_mapping)
    print(f"  Half GPU/Half QPU: {cost_split:.0f} bytes (cross-backend transfers)")
    
    # Scenario C: Alternating backends (worst case)
    alt_mapping = {i: ('GPU' if i % 2 == 0 else 'QPU') for i in range(16)}
    cost_alt = tg.cross_backend_cost(alt_mapping)
    print(f"  Alternating:       {cost_alt:.0f} bytes (worst case)")
    
    assert cost_same == 0, "Same-backend should have 0 transfer cost"
    assert cost_split > 0, "Split-backend should have non-zero cost"
    assert cost_alt >= cost_split, "Alternating should be >= split cost"
    print("  PASSED!")
    
    # --- TEST 6: Graph properties ---
    print("\n--- TEST 6: Graph Properties ---")
    print(f"  Is DAG:            {nx.is_directed_acyclic_graph(tg.graph)}")
    print(f"  Connected:         {nx.is_weakly_connected(tg.graph)}")
    
    in_degrees = dict(tg.graph.in_degree())
    out_degrees = dict(tg.graph.out_degree())
    
    # Membrane tasks: 0 in-degree (sources), high out-degree
    mem_in = [in_degrees[i*2] for i in range(8)]
    mem_out = [out_degrees[i*2] for i in range(8)]
    # Synaptic tasks: high in-degree (sinks), 0 out-degree
    syn_in = [in_degrees[i*2+1] for i in range(8)]
    syn_out = [out_degrees[i*2+1] for i in range(8)]
    
    print(f"  Membrane in-degree:  {mem_in[0]} (sources - no dependencies)")
    print(f"  Membrane out-degree: {mem_out[0]} (feeds 1 own synapse + 7 others = 8)")
    print(f"  Synaptic in-degree:  {syn_in[0]} (receives from 1 own + 7 others = 8)")
    print(f"  Synaptic out-degree: {syn_out[0]} (sinks - no dependents)")
    
    assert all(d == 0 for d in mem_in), "Membrane tasks should have 0 in-degree"
    assert all(d == 8 for d in mem_out), "Membrane tasks should feed 8 tasks"
    assert all(d == 8 for d in syn_in), "Synaptic tasks should receive from 8 tasks"
    assert all(d == 0 for d in syn_out), "Synaptic tasks should have 0 out-degree"
    print("  PASSED!")
    
    # --- TEST 7: Visualize with mock backend mapping ---
    print("\n--- TEST 7: Visualization ---")
    # Simulate a realistic mapping (mix of backends)
    mock_mapping = {}
    for i in range(16):
        task = tg.tasks[i]
        if task.task_type == 'membrane' and task.neuron_id < 3:
            mock_mapping[i] = 'QPU'     # First 3 neurons' membrane on QPU
        elif task.task_type == 'membrane':
            mock_mapping[i] = 'GPU'     # Rest of membrane on GPU
        elif task.neuron_id < 3:
            mock_mapping[i] = 'GPU'     # QPU neurons' synapses on GPU
        else:
            mock_mapping[i] = 'CPU'     # Rest of synapses on CPU
    
    save_path = os.path.join(session.figures_dir, 'week3_task_graph.png')
    tg.visualize(mock_mapping, save_path=save_path)
    
    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Tasks:        {len(tg.tasks)} (8 membrane + 8 synaptic)")
    print(f"  Edges:        {tg.graph.number_of_edges()} (8 intra + 56 cross)")
    print(f"  A matrix:     {A.shape} with {np.sum(A)} non-zero entries")
    print(f"  D matrix:     {D.shape}, total {np.sum(D):.0f} bytes")
    print(f"  Is DAG:       {nx.is_directed_acyclic_graph(tg.graph)}")
    print(f"  Graph saved:  {save_path}")
    print()
    print("  KEY INSIGHT: The task graph shows that membrane tasks are")
    print("  SOURCES (no dependencies) and synaptic tasks are SINKS")
    print("  (depend on all membrane tasks). This bipartite structure")
    print("  means ALL membrane tasks can run in PARALLEL, then all")
    print("  synaptic tasks can run in parallel - perfect for GPU batch!")
    print("=" * 60)
