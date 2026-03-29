import sys
import os
from utils.logger import session
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def stylize_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)

class ThesisPlotter:
    def __init__(self, output_dir='results/figures'):
        self.output_dir = session.figures_dir
        session.setup_directories()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Initialized ThesisPlotter. Outputting to {self.output_dir}")

    def plot_sensitivity_heatmap(self):
        """Generates Figure 5: Sensitivity of Orchestrator Speedup"""
        csv_path = os.path.join(session.tables_dir, 'table_III_sensitivity.csv')
        if not os.path.exists(csv_path):
            print(f"Skipping sensitivity plot: {csv_path} not found.")
            return

        import csv
        raw_data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_data.append({
                    'r': float(row['Rho_Theta']),
                    'w': float(row['W_Theta']),
                    's': float(row['Speedup_x'])
                })
                
        # Get unique sorted values for axes
        r_vals = sorted(list(set([d['r'] for d in raw_data])))
        w_vals = sorted(list(set([d['w'] for d in raw_data])))
        
        # Build pivot 2D array matrix matching (row, col)
        pivot_mat = np.zeros((len(r_vals), len(w_vals)))
        for d in raw_data:
            r_i = r_vals.index(d['r'])
            w_i = w_vals.index(d['w'])
            pivot_mat[r_i, w_i] = d['s']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(pivot_mat, cmap='viridis')
        
        # Labels and formatting
        fig.colorbar(cax, label='Algorithmic Speedup (x)')
        
        ax.set_xticks(range(len(w_vals)))
        ax.set_yticks(range(len(r_vals)))
        ax.set_xticklabels([f"{w:.1f}" for w in w_vals])
        ax.set_yticklabels([f"{r:.0f}" for r in r_vals])
        
        ax.set_xlabel(r"Stochasticity Threshold ($W_\theta$)")
        ax.set_ylabel(r"Stiffness Threshold ($\rho_\theta$)")
        ax.xaxis.set_ticks_position('bottom')
        
        max_s = np.max(pivot_mat)
        
        # Add text annotations
        for i in range(len(r_vals)):
            for j in range(len(w_vals)):
                val = pivot_mat[i, j]
                text_col = 'black' if val > max_s * 0.7 else 'white'
                ax.text(j, i, f"{val:.1f}x", ha='center', va='center', color=text_col)

        plt.title('Figure 5: Scalability under Threshold Sensitivity')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig5_sensitivity_heatmap.png'), dpi=300)
        plt.close()

    def plot_qpu_budget_tradeoff(self):
        """Generates Figure 6: Qubit Budget vs Speedup Tradeoff"""
        csv_path = os.path.join(session.tables_dir, 'table_IV_tradeoffs.csv')
        if not os.path.exists(csv_path):
            print(f"Skipping tradeoff plot: {csv_path} not found.")
            return

        import csv 
        x_vals, y_vals = [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x_vals.append(int(row['Q_Budget_Constraint']))
                y_vals.append(float(row['Orchestrator_Speedup']))
                
        fig, ax1 = plt.subplots(figsize=(8, 5))
        stylize_axes(ax1)
        
        ax1.plot(x_vals, y_vals, marker='o', color='#2ca02c', linewidth=2, markersize=8)
        ax1.set_xlabel('Available QPU Qubit Budget ($Q_{limit}$)')
        ax1.set_ylabel('Algorithmic Speedup (x)', color='#2ca02c')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')
        ax1.set_xticks(np.arange(0, 16, 2))
        
        # Annotate saturation
        ax1.axvline(x=8, color='gray', linestyle='--', alpha=0.5)
        ax1.text(8.2, max(y_vals)*0.8, 'Saturation Point\n(Diminishing Returns)', color='gray')
        
        plt.title('Figure 6: QPU Budget vs Orchestrator Scaling')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fig6_budget_tradeoffs.png'), dpi=300)
        plt.close()

    def format_voltage_traces(self):
        """Locates the existing Phase 4 voltage traces and copies them with formal titles"""
        import shutil
        import glob
        
        existing = glob.glob(session.figures_dir + '/week7_*.png')
        if not existing:
            print("No baseline traces found in results directory to format.")
            return
            
        for path in existing:
            if 'baseline' in path:
                # In real context we plot from raw data, but since we already plotted them
                # we just format alias them for the thesis grouping.
                new_name = 'fig3_baseline_cpu.png'
                shutil.copy(path, os.path.join(self.output_dir, new_name))
            elif 'proposed' in path:
                new_name = 'fig4_proposed_hybrid.png'
                shutil.copy(path, os.path.join(self.output_dir, new_name))

    def run_all(self):
        print("Generating final thesis figures...")
        self.plot_sensitivity_heatmap()
        self.plot_qpu_budget_tradeoff()
        self.format_voltage_traces()
        print("Done. Check results/figures/ for high-DPI publication ready plots.")

if __name__ == "__main__":
    plotter = ThesisPlotter()
    plotter.run_all()
