"""
Week 6: Eigenvector Continuation (Algorithm 3)

WHAT IS EIGENVECTOR CONTINUATION (EC)?
──────────────────────────────────────
Normally, to find the stiffness index ρ of a neuron, we must diagonalize 
its Jacobian matrix. A full eigenvalue solver has O(N³) complexity. For large
coupled systems, this becomes a major bottleneck.

EC is a Quantum-inspired mathematical trick to bypass this O(N³) solve.

HOW IT WORKS:
─────────────
1. OFFLINE (Anchor Solving):
   - At startup, we solve the exact eigenvalue problem for the Jacobian
     at a few "anchor points" (e.g., different resting/spiking states).
   - We extract their eigenvectors and stack them to form a subspace V.

2. ONLINE (Target Prediction):
   - When we need ρ for a new state, we don't diagonalize its big Jacobian J.
   - We PROJECT the big Jacobian onto the small subspace: J_proj = V^T * J * V 
   - We solve the tiny J_proj eigenvalue problem.
   - Complexity drops dramatically in exchange for ~1% approximation error.

Paper reference:
- Algorithm 3
- Equation 8: H_proj * c = lambda * N * c
"""

import numpy as np
import time

class EigenvectorContinuation:
    """
    Implements fast stiffness (eigenvalue) approximation using subspace projection.
    """
    def __init__(self, basis_size=4):
        self.basis_size = basis_size
        self.V = None  # The orthonormal basis matrix (N x M)
        self.is_trained = False

    def offline_train(self, anchor_jacobians):
        """
        Takes a list of 'anchor' Jacobians evaluated at different 
        representative states, solves them exactly, and builds the
        subspace basis V.
        
        Args:
            anchor_jacobians: List of full NxN numpy arrays.
        """
        # print(f"  [EC] Starting offline training with {len(anchor_jacobians)} anchor matrices...")
        
        all_eigenvectors = []
        for J in anchor_jacobians:
            # Exact slow solve for anchors
            eigenvalues, eigenvectors = np.linalg.eig(J)
            
            # Extract real and imaginary parts separately to keep V strictly real-valued.
            # This is a common numerical trick when projecting dynamical systems.
            for i in range(eigenvectors.shape[1]):
                v = eigenvectors[:, i]
                all_eigenvectors.append(v.real)
                if np.any(np.abs(v.imag) > 1e-10):
                    all_eigenvectors.append(v.imag)
                    
        # Stack into a massive matrix (N x many)
        V_raw = np.column_stack(all_eigenvectors)
        
        # We want an orthonormal basis to avoid numerical instability
        # QR decomposition gives us an orthonormal basis Q
        Q, R = np.linalg.qr(V_raw)
        
        # Truncate to the desired basis size (or max rank available)
        keep = min(self.basis_size, Q.shape[1])
        self.V = Q[:, :keep]
        self.is_trained = True
        
        # print(f"  [EC] Training complete. Subspace V shape: {self.V.shape}")

    def predict_eigenvalues(self, J_target):
        """
        Online Stage (Algorithm 3 prediction)
        Instead of O(N³) full solve, project J_target down to O(r³).
        
        Args:
            J_target: The (N x N) Jacobian to approximate.
            
        Returns:
            Approximate eigenvalues.
        """
        if not self.is_trained:
            raise RuntimeError("Eigenvector Continuation is not trained!")
            
        # 1. Project the large NxN matrix onto the tiny rxr subspace
        # J_proj = V^T * J_target * V
        J_proj = self.V.T @ J_target @ self.V  # Geometry: (r x N) * (N x N) * (N x r) = (r x r)
        
        # 2. Solve the tiny (r x r) problem
        # Because we orthonormalized V via QR (V^T*V = I), the overlap 
        # matrix N from Eq 8 is the Identity matrix. So we solve standard eig!
        approx_eigenvalues, _ = np.linalg.eig(J_proj)
        
        return approx_eigenvalues

    def compute_stiffness_rho(self, J_target):
        """
        Wrapper to directly return exactly what the Orchestrator needs: 
        the stiffness index rho.
        """
        approx_evals = self.predict_eigenvalues(J_target)
        abs_evals = np.abs(np.real(approx_evals))
        abs_evals = abs_evals[abs_evals > 1e-10]  # Filter out exact zeros
        
        if len(abs_evals) == 0:
            return 1.0  # Safe default if matrix is null
            
        rho = np.max(abs_evals) / np.min(abs_evals)
        return float(rho)


# ===================================================================
#  SELF-TEST
# ===================================================================
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from core.hh_neuron import hh_derivatives
    from core.metrics import DynamicsMetrics
    
    print("=" * 60)
    print("WEEK 6: Eigenvector Continuation (Algorithm 3) — Verification")
    print("=" * 60)
    
    metrics = DynamicsMetrics()
    ec = EigenvectorContinuation(basis_size=4)
    
    # Generate 4 "Anchor" states points across a single spike setup
    np.random.seed(42)
    anchor_states = [
        np.array([-65.0, 0.05, 0.6, 0.32]),   # Resting
        np.array([-40.0, 0.5, 0.4, 0.4]),     # Depolarizing
        np.array([ 30.0, 0.99, 0.05, 0.5]),   # Spike peak
        np.array([-75.0, 0.01, 0.8, 0.2])     # Hyperpolarization
    ]
    
    print("--- TEST 1: Offline Training Stage ---")
    t0 = time.time()
    anchor_Js = [metrics.compute_jacobian(hh_derivatives, state, 0) for state in anchor_states]
    ec.offline_train(anchor_Js)
    train_time = (time.time() - t0) * 1000
    print(f"  Trained carefully on {len(anchor_Js)} exact Jacobians.")
    print(f"  Training Wallclock: {train_time:.2f} ms")
    print(f"  Orthonormal Basis V shape: {ec.V.shape}")
    assert ec.is_trained
    print("  PASSED!")
    
    
    print("\n--- TEST 2: Online Prediction (Unseen Target Jacobian) ---")
    # A random state that was NOT in the anchor training data
    target_state = np.array([-10.0, 0.75, 0.2, 0.45])
    J_target = metrics.compute_jacobian(hh_derivatives, target_state, 0)
    
    # 1. Compute EXACT the slow way
    t_start = time.time()
    exact_evals, _ = np.linalg.eig(J_target)
    exact_rho = metrics.stiffness_index(J_target)
    exact_time = time.time() - t_start
    
    # 2. Compute APPROXIMATE the fast EC way
    t_start = time.time()
    approx_evals = ec.predict_eigenvalues(J_target)
    approx_rho = ec.compute_stiffness_rho(J_target)
    approx_time = time.time() - t_start
    
    # Compare
    # Sort eigenvalues to compare magnitudes directly
    exact_sorted = np.sort(np.abs(np.real(exact_evals)))
    approx_sorted = np.sort(np.abs(np.real(approx_evals)))
    
    print(f"  Exact  Eigenvalues (Real): {exact_sorted}")
    print(f"  Approx Eigenvalues (Real): {approx_sorted}")
    
    print(f"\n  Exact Stiffness ρ:  {exact_rho:8.2f}  (Time: {exact_time*1000:6.3f} ms)")
    print(f"  Approx Stiffness ρ: {approx_rho:8.2f}  (Time: {approx_time*1000:6.3f} ms)")
    
    # Speedup and Accuracy
    rho_error = abs(exact_rho - approx_rho) / exact_rho
    speedup = exact_time / approx_time if approx_time > 0 else float('inf')
    
    print(f"\n  Accuracy Error: {rho_error:.2%}")
    print(f"  Speedup:        {speedup:.1f}x")
    
    assert rho_error < 0.15, "Error exceeds 15% tolerance!"
    print("  PASSED!")
    print("============================================================")
