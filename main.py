import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.logger import session, TerminalColors as tc

def print_banner():
    print(f"\n{tc.HEADER}{tc.BOLD}" + "=" * 65)
    print("   INTELLIGENT ORCHESTRATOR FOR HYBRID BRAIN SIMULATION (v1.0)")
    print("=" * 65 + f"{tc.ENDC}")
    print(f"{tc.OKCYAN}Current Session: {tc.BOLD}{session.timestamp}{tc.ENDC}")
    print(f"{tc.OKCYAN}Output Folder:   {tc.BOLD}results/runs/{session.timestamp}/{tc.ENDC}\n")

def display_menu():
    print(f"{tc.BOLD}AVAILABLE EXPERIMENT MODULES:{tc.ENDC}")
    print(f"  {tc.OKBLUE}[1]{tc.ENDC} Run Core Module: HH Neuron Sandbox {tc.OKCYAN}(Week 1){tc.ENDC}")
    print(f"  {tc.OKBLUE}[2]{tc.ENDC} Run Core Module: Dynamics Metrics {tc.OKCYAN}(Week 2){tc.ENDC}")
    print(f"  {tc.OKBLUE}[3]{tc.ENDC} Run Core Module: Neural Task Graph {tc.OKCYAN}(Week 3){tc.ENDC}")
    print(f"  {tc.OKBLUE}[4]{tc.ENDC} Run Core Module: Algorithm 1 (Partitioning) {tc.OKCYAN}(Week 4){tc.ENDC}")
    print(f"  {tc.OKBLUE}[5]{tc.ENDC} Run Core Module: ML Surrogate ReBaNO Simulator {tc.OKCYAN}(Week 5){tc.ENDC}")
    print(f"  {tc.OKBLUE}[6]{tc.ENDC} Run Core Module: SD-Policy Allocation Verifier {tc.OKCYAN}(Week 6){tc.ENDC}\n")
    
    print(f"{tc.BOLD}THESIS RESULTS GENERATOR:{tc.ENDC}")
    print(f"  {tc.OKGREEN}[7]{tc.ENDC} Run Baseline Sequence Evaluation {tc.OKCYAN}(Phase 4){tc.ENDC}")
    print(f"  {tc.OKGREEN}[8]{tc.ENDC} Run Proposed Hybrid Quantum Evaluation {tc.OKCYAN}(Phase 4){tc.ENDC}")
    print(f"  {tc.OKGREEN}[9]{tc.ENDC} Benchmark: Orchestrator Threshold Sensitivities {tc.OKCYAN}(Phase 5){tc.ENDC}")
    print(f"  {tc.OKGREEN}[10]{tc.ENDC} Benchmark: Hardware Q_Budget Tradeoffs {tc.OKCYAN}(Phase 5){tc.ENDC}")
    print(f"  {tc.OKGREEN}[11]{tc.ENDC} Compile Final Thesis Academic Plots {tc.OKCYAN}(Phase 5){tc.ENDC}\n")
    
    print(f"{tc.BOLD}AUTOMATION:{tc.ENDC}")
    print(f"  {tc.WARNING}[A]{tc.ENDC} Run ALL Modules sequentially (1 -> 11)\n")
    
    print(f"  {tc.FAIL}[X]{tc.ENDC} Exit Terminal\n")

def run_script(script_path):
    # Dynamically inject the session folder mapping if running natively
    session.setup_directories()
    
    if not os.path.exists(script_path):
        session.log_error(f"Could not locate {script_path}. Is the file moved?")
        return

    session.log_info(f"Triggering Execution -> {script_path} ...")
    print(f"{tc.WARNING}{'-'*65}{tc.ENDC}")
    
    start_t = time.time()
    try:
        # Pass PYTHONPATH so all scripts can resolve absolute 'utils.logger' imports
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
        
        # Popen inherently pipes colors directly into standard output as it renders
        subprocess.run([sys.executable, script_path], check=True, env=env)
        dur = time.time() - start_t
        print(f"{tc.WARNING}{'-'*65}{tc.ENDC}")
        session.log_success(f"Execution completed in {dur:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"{tc.WARNING}{'-'*65}{tc.ENDC}")
        session.log_error(f"Execution failed with return code {e.returncode}.")
    except Exception as e:
        print(f"{tc.WARNING}{'-'*65}{tc.ENDC}")
        session.log_error(f"Unhandled Exception: {str(e)}")

def main_loop():
    # Enforce standard execution path
    base = os.path.dirname(os.path.abspath(__file__))
    
    options = {
        '1': os.path.join(base, 'core', 'hh_neuron.py'),
        '2': os.path.join(base, 'core', 'metrics.py'),
        '3': os.path.join(base, 'core', 'task_graph.py'),
        '4': os.path.join(base, 'algorithms', 'partitioning.py'),
        '5': os.path.join(base, 'algorithms', 'rebano.py'),
        '6': os.path.join(base, 'algorithms', 'sd_policy.py'),
        '7': os.path.join(base, 'experiments', 'run_baseline.py'),
        '8': os.path.join(base, 'experiments', 'run_proposed.py'),
        '9': os.path.join(base, 'experiments', 'run_sensitivity.py'),
        '10': os.path.join(base, 'experiments', 'run_tradeoffs.py'),
        '11': os.path.join(base, 'visualization', 'plotters.py')
    }
    
    while True:
        print_banner()
        display_menu()
        
        choice = input(f"{tc.BOLD}Select an operation [1-11, A, or X]: {tc.ENDC}").strip().upper()
        
        if choice == 'X':
            session.log_info("Terminating Orchestrator. Session data saved.")
            break
            
        if choice == 'A':
            for i in range(1, 12):
                run_script(options[str(i)])
            input(f"\n{tc.BOLD}All 11 modules successfully executed! Press Enter to return to main menu...{tc.ENDC}")
        elif choice in options:
            run_script(options[choice])
            input(f"\n{tc.BOLD}Press Enter to return to main menu...{tc.ENDC}")
        else:
            session.log_error("Invalid selection. Please input a number from 1 to 11, 'A', or 'X'.")
            input(f"\n{tc.BOLD}Press Enter to return to main menu...{tc.ENDC}")

if __name__ == "__main__":
    # OS color validation block explicitly requesting ansi pass-throughs
    if os.name == 'nt':
        os.system('color')
        
    main_loop()
