"""
Week 1: Hodgkin-Huxley Neuron Model

The HH model describes how neurons generate action potentials (voltage spikes)
using 4 coupled differential equations for:
  V = membrane voltage (mV)
  m = sodium channel activation gate (0 to 1)
  h = sodium channel inactivation gate (0 to 1)  
  n = potassium channel activation gate (0 to 1)

This is the FOUNDATION of the entire project — every other module
depends on this neuron model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def hh_derivatives(t, state, Iext=10.0, noise=0.0):
    """
    The 4 Hodgkin-Huxley equations.
    state = [V, m, h, n]
    """
    V, m, h, n = state
    
    # HH parameters
    Cm  = 1.0;  gNa = 120.0; gK = 36.0;  gL = 0.3
    ENa = 50.0; EK  = -77.0; EL = -54.4
    
    # Gate rate functions
    am = 0.1*(V+40)/(1-np.exp(-(V+40)/10)) if V != -40 else 1.0
    bm = 4*np.exp(-(V+65)/18)
    ah = 0.07*np.exp(-(V+65)/20)
    bh = 1/(1+np.exp(-(V+35)/10))
    an = 0.01*(V+55)/(1-np.exp(-(V+55)/10)) if V != -55 else 0.1
    bn = 0.125*np.exp(-(V+65)/80)
    
    # Currents
    INa = gNa * m**3 * h * (V - ENa)
    IK  = gK  * n**4     * (V - EK)
    IL  = gL             * (V - EL)
    
    # 4 differential equations
    dVdt = (Iext + noise - INa - IK - IL) / Cm
    dmdt = am*(1-m) - bm*m
    dhdt = ah*(1-h) - bh*h
    dndt = an*(1-n) - bn*n
    
    return [dVdt, dmdt, dhdt, dndt]


if __name__ == "__main__":
    import os
    from utils.logger import session
    print("Starting simulation for 1 Hodgkin-Huxley Neuron...")
    
    # Solve and plot
    t_span  = (0, 100)                        # 0 to 100 ms
    t_eval  = np.linspace(0, 100, 4000)       # 4000 time points
    y0      = [-65.0, 0.05, 0.6, 0.32]        # resting state
    
    sol = solve_ivp(hh_derivatives, t_span, y0,
                    t_eval=t_eval, method='RK45',
                    rtol=1e-6, atol=1e-8)
                    
    # Plot all 4 variables
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    labels = ['V (mV)', 'm (Na act)', 'h (Na inact)', 'n (K act)']
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(sol.t, sol.y[i], color=color, linewidth=1.5)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_title('Hodgkin-Huxley Neuron - Single Spike', 
                      fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save to dynamic session directory
    out_dir = session.figures_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'hh_neuron_output.png')
    plt.savefig(out_path, dpi=150)
    print(f"Success! Check {out_path}")
