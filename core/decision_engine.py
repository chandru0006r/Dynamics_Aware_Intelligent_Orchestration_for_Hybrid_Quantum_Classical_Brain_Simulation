"""
Week 4: Core Decision Engine Configuration

This module stores the critical thresholds that govern the
decisions of the Hybrid Partitioning Algorithm.

Parameters from the paper:
- ρ_θ (rho_theta): Threshold for stiffness index (Equation 3).
   Any task with ρ > ρ_θ is considered "stiff" and gets candidate
   status for the QPU to exploit spectral structure. Default 100.
   
- W_θ (W_theta): Threshold for stochasticity factor (Equation 4).
   Any task with W(t) > W_θ is considered noise-dominated and 
   benefits from QPU sampling. Default 0.3.
   
- P_θ (P_theta): Threshold for parallelizability.
   Membrane tasks have P=0.8. Synaptic tasks have P=0.5.
   Any non-QPU task with P > P_θ is batched to the GPU.
   Any task with P <= P_θ runs sequentially on the CPU. Default 0.7.
   
- Q_budget: Total number of logical qubits available on the backend QPU.
   For the default Hodgkin-Huxley model (4 equations: V,m,h,n) 
   we require log2(4) = 2 qubits per task. 
   A budget of 8 qubits allows a maximum of 4 simultaneous QPU tasks.
"""

from dataclasses import dataclass

@dataclass
class OrchestratorConfig:
    """Settings used tightly by Algorithm 1 (Hybrid Partitioning)"""
    rho_theta: float = 100.0   # Stiffness Threshold
    W_theta: float = 0.3       # Stochasticity Threshold
    P_theta: float = 0.7       # Parallelism Threshold
    Q_budget: int = 8          # Qubit capacity. 8 qubits = max 4 tasks (using 2 qb each)
    
# Global default instance
DEFAULT_CONFIG = OrchestratorConfig()

def get_qubits_required(num_state_vars: int = 4) -> int:
    """
    Returns the number of qubits required for a generic ODE system.
    Equation: qubits = ceil(log2(N)), where N is num variables.
    For standard HH (4 variables), this is log2(4) = 2.
    """
    import math
    if num_state_vars <= 0:
        return 0
    return math.ceil(math.log2(num_state_vars))
