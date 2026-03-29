# Dynamics-Aware Intelligent Orchestration for Hybrid Quantum-Classical Brain Simulation

This repository contains the full programmatic implementation of the Final Year Thesis: **Intelligent Orchestrator for Hybrid Brain Simulation**. 

The system leverages a dynamically scaling heterogeneous pipeline that accurately monitors the execution states of `Hodgkin-Huxley` biological equations. It selectively routes "stiff" mathematical problems away from Classical CPU solvers and towards Quantum Surrogate Neural proxies, demonstrating **up to ~3.7x Algorithmic Speedup**.

## 1. Project Navigation
The core architecture is encapsulated in `implementation/`:
* `algorithms/` — ReBaNO Surrogate Quantum solver & the Spectral Clustering Partitioners.
* `backends/` — Hardware execution environment wrappers (CPU, GPU, QPU).
* `core/` — The physics models (`hh_neuron.py`) and standard mathematical tracking (`metrics.py`).
* `experiments/` — Scaling simulators for Sensitivity & Hardware Tradeoff datasets.
* `main.py` — The unified execution Interface.

## 2. Requirements & Installation
Ensure you are running **Python 3.10+**. The system relies entirely on standard mathematical computation modules (`numpy`, `scipy`) for pure execution fidelity and `matplotlib` for generating the graphs.

```bash
# Clone the repository
git clone https://github.com/chandru0006r/Dynamics_Aware_Intelligent_Orchestration_for_Hybrid_Quantum_Classical_Brain_Simulation.git
cd Dynamics_Aware_Intelligent_Orchestration_for_Hybrid_Quantum_Classical_Brain_Simulation

# Install Python requirements
pip install -r requirements.txt
```

## 3. How to Run the Thesis Engine
Instead of wrestling with manual script execution, the whole project has been unified into a sleek, colorful Interactive Terminal Menu.

```bash
cd implementation
python main.py
```

### Navigating the Engine:
Once launched, you'll be greeted by an interactive menu covering all weeks of development:

- **[1 - 6] Component Validations**: Execute individual isolated checks (e.g. Test if the Neuron correctly spikes, evaluate the SD-Policy, test metric calculations).
- **[7 - 11] Phase 4/5 Experiment Suites**: Run the massive, end-to-end multi-millisecond network tests. These modules extract tables, calculate QPU budgets, and test threshold sensitivity equations automatically.
- **[A] Run All Modules**: Pressing `A` will sequentially execute every single algorithm module (1 through 11).

> **Note on Outputs:** 
> Executing `main.py` generates a unique Session Timestamp (e.g. `2026-03-29_23-50-01`). Every visual output graph, plot, or table exported during that runtime is saved automatically inside isolated directories beneath `implementation/results/runs/`. You will never accidentally overwrite previous mathematical data!

## 4. Academic Data Sets
The data mapping directly corresponds to tables and figures provided in the thesis manuscript documentation:
* **Fig 3 & 4**: Voltage Trace Accuracy validation.
* **Fig 5**: Algorithmic speedup sensitivity heatmap matrix.
* **Fig 6**: QPU Resource Budget Scaling tradeoff.
* **Table III & IV**: Algorithmic data tables evaluating stochastic tolerances.
