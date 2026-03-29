"""
Week 6: Slowdown-Driven Scheduling Policy (Algorithm 4)

WHAT IS SD-POLICY?
──────────────────
The Hybrid Partitioner (Algorithm 1) simply bins tasks into CPU, GPU, 
and QPU groups. But what if 1000 tasks are sent to the GPU and bottleneck it?

The Slowdown-Driven (SD) Policy is a malleable resource scheduler.
It monitors the expected latency (Slowdown, SD) of each task.
If SD exceeds a threshold (SD_θ), the scheduler forcefully shrinks
the resources allocated to low-stiffness tasks (reducing their priority)
to ensure the high-stiffness tasks stay under the latency threshold.

Formula:
  SD = execution_time_shared / execution_time_isolated
  SD = 1 / resource_fraction
  
If SD > SD_θ, we iteratively penalize the lowest ρ tasks.

Paper reference:
- Algorithm 4 (SD-Policy Iterative Loop)
"""

class SDPolicyScheduler:
    """
    Implements malleable scheduling based on the Slowdown metric.
    """
    def __init__(self, sd_theta=1.5):
        """
        Args:
            sd_theta: The max allowable slowdown for high-priority tasks.
                      1.5 means a task can run 50% slower than isolated 
                      before the scheduler kicks in and penalizes others.
        """
        self.sd_theta = sd_theta
        
        # Log to track scheduling actions
        self.history_log = []

    def schedule_queue(self, task_queue, backend_name, total_resources=1.0):
        """
        Applies SD-Policy to a specific backend's queue of tasks.
        
        Args:
            task_queue: A list of dicts. Ex: [{'task_id': 0, 'rho': 150.0}, ...]
            backend_name: String for logging ('GPU', 'QPU', etc.)
            total_resources: 1.0 represents 100% of the backend's bandwidth.
            
        Returns:
            allocated_queue: The queue updated with 'fraction' and 'sd'.
        """
        if len(task_queue) == 0:
            return []
            
        # 1. Base sharing (Fair-share scheduling)
        # If 4 tasks share a GPU, they each get 0.25 of the resources.
        base_fraction = total_resources / len(task_queue)
        sd_current = 1.0 / base_fraction
        
        # Apply base fractions initially
        for task in task_queue:
            task['fraction'] = base_fraction
            task['sd'] = sd_current
            
        # 2. Threshold Check
        if sd_current <= self.sd_theta:
            # Everything is fine. Fair-share is within latency SLAs.
            self.history_log.append(f"[{backend_name}] Queue length {len(task_queue)} OK. SD={sd_current:.2f}")
            return task_queue
            
        # 3. Violation! Apply Algorithm 4 Malleability
        self.history_log.append(f"[{backend_name}] VIOLATION! SD={sd_current:.2f} > {self.sd_theta}. Shrinking low-priority tasks.")
        
        # Sort tasks by stiffness ρ (highest first)
        sorted_tasks = sorted(task_queue, key=lambda x: x.get('rho', 0), reverse=True)
        
        # We want to give the top priority tasks enough resources so their SD = SD_θ
        # target_fraction = 1.0 / SD_θ
        target_fraction = 1.0 / self.sd_theta
        
        # How many high-priority tasks can we fit at this target_fraction?
        max_high_prio_count = int(total_resources // target_fraction)
        
        # We must leave some tiny fraction for the low-priority tasks so they don't starve
        min_survival_fraction = 0.05
        
        resources_remaining = total_resources
        
        for i, task in enumerate(sorted_tasks):
            # Is this a high priority task?
            if i < max_high_prio_count:
                # Give it exactly what it needs to hit the latency target
                allocation = min(target_fraction, resources_remaining - ((len(sorted_tasks) - i - 1) * min_survival_fraction))
            else:
                # It's a low priority task. Penalize it heavily to soak up the overflow.
                tasks_remaining = len(sorted_tasks) - i
                allocation = resources_remaining / tasks_remaining
                
            task['fraction'] = allocation
            task['sd'] = 1.0 / allocation if allocation > 0 else float('inf')
            resources_remaining -= allocation
            
            # Record decision
            class_str = "HIGH-PRIO" if i < max_high_prio_count else "SHRUNK"
            self.history_log.append(
                f"  -> Task {task['task_id']:>2} (ρ={task['rho']:>6.1f}): {class_str} assigned {allocation*100:>4.1f}% resources (SD={task['sd']:.2f})"
            )
            
        return sorted_tasks

    def apply_batch(self, mapping_dict, task_graph):
        """
        Orchestrator interface. Takes the raw Algorithm 1 output and applies
        the SD policy across all backends simultaneously.
        """
        # Bin tasks
        queues = {'CPU': [], 'GPU': [], 'QPU': []}
        for tid, backend in mapping_dict.items():
            # Get stiffness (default 0 for synaptic tasks)
            t_rho = getattr(task_graph.tasks[tid], 'rho', 0.0)
            queues[backend].append({'task_id': tid, 'rho': t_rho})
            
        # Apply policies
        final_allocations = {}
        for backend, q in queues.items():
            result_q = self.schedule_queue(q, backend, total_resources=1.0)
            for t in result_q:
                final_allocations[t['task_id']] = {
                    'backend': backend,
                    'fraction': t['fraction'],
                    'sd': t['sd']
                }
                
        return final_allocations

# ===================================================================
#  SELF-TEST & VISUALIZATION
# ===================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    from utils.logger import session
    print("=" * 60)
    print("WEEK 6: SD-Policy Scheduler (Algorithm 4) — Verification")
    print("=" * 60)
    
    # Let's simulate a full System Batch of 16 tasks across CPU, GPU, QPU
    # CPU: 8 Synaptic Tasks (low rho, uncongested limit)
    cpu_queue = [
        {'task_id': i, 'rho': 0.0} for i in range(1, 9)
    ]
    # GPU: 4 Membrane Tasks (highly congested limit)
    gpu_queue = [
        {'task_id': 10, 'rho': 90.0},  # SLA: keep this one fast!
        {'task_id': 11, 'rho': 80.0},
        {'task_id': 12, 'rho': 20.0},
        {'task_id': 13, 'rho': 10.0}
    ]
    # QPU: 4 Stiff Membrane Tasks (uncongested QPU)
    qpu_queue = [
        {'task_id': 20, 'rho': 300.0},
        {'task_id': 21, 'rho': 250.0},
        {'task_id': 22, 'rho': 150.0},
        {'task_id': 23, 'rho': 120.0}
    ]

    # CPU can handle many small tasks, so SD_theta is high
    sched_cpu = SDPolicyScheduler(sd_theta=10.0)
    res_cpu = sched_cpu.schedule_queue(cpu_queue, 'CPU')

    # GPU is overloaded. We strictly enforce SD_theta=1.5 to protect the stiffest task
    sched_gpu = SDPolicyScheduler(sd_theta=1.5)
    res_gpu = sched_gpu.schedule_queue(gpu_queue, 'GPU')

    # QPU handles specific 2-qubit slices smoothly
    sched_qpu = SDPolicyScheduler(sd_theta=5.0)
    res_qpu = sched_qpu.schedule_queue(qpu_queue, 'QPU')

    print("\n--- TEST 1: System-Wide Output ---")
    for log in sched_gpu.history_log:
        print("  " + log)

    # Combine results
    all_results = [
        ('CPU', '#CCCCCC', res_cpu),
        ('GPU', '#4285F4', res_gpu),
        ('QPU', '#EA4335', res_qpu)
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = 0
    y_ticks = []
    y_labels = []

    for backend, color, queue in reversed(all_results):
        for task in reversed(queue):
            pct = task['fraction'] * 100
            rho = task.get('rho', 0.0)
            
            # Print label details dynamically
            label = f"T{task['task_id']} (ρ={rho:.0f})" if backend != 'CPU' else f"T{task['task_id']} (Synapse)"
            
            # Draw Horizontal Bar
            ax.barh(y_pos, pct, color=color, edgecolor='black', height=0.7)
            
            # Text inside bar
            ax.text(pct + 1, y_pos, f"{pct:.1f}%", va='center', fontweight='bold', fontsize=9)
            
            y_ticks.append(y_pos)
            y_labels.append(label)
            y_pos += 1
            
        y_pos += 1  # Add gap between backends

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Resource Allocation Percentage (%)')
    ax.set_title('Week 6 Output: SD-Policy Dynamic Resource Scaling', fontweight='bold')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EA4335', edgecolor='black', label='QPU (Stiff Tasks)'),
        Patch(facecolor='#4285F4', edgecolor='black', label='GPU (Congested: Shrunk Low-Priority!)'),
        Patch(facecolor='#CCCCCC', edgecolor='black', label='CPU (Uncongested Synapses)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_dir = session.figures_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'week6_sdpolicy_output.png')
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved successfully to: {out_path}")
    print("============================================================")
