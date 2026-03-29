import os
import glob

base_dir = r"c:\Users\KEERTHIVASAN S\Downloads\Intelligent Orchestrator for Hybrid Brain Simulation\implementation"
files_to_patch = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith('.py') and f not in ['logger.py', 'patch_logging.py', 'main.py', '__init__.py']:
            files_to_patch.append(os.path.join(root, f))

for file_path in files_to_patch:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip files that don't output anything
    if 'results' not in content and 'figures' not in content and 'tables' not in content:
        continue

    # Add the session import right after 'import os'
    if 'from utils.logger import session' not in content:
        content = content.replace('import os\n', 'import os\nfrom utils.logger import session\n', 1)

    # Patch plotting lines based on what we see in the greps
    content = content.replace("out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')", "out_dir = session.figures_dir")
    content = content.replace("out_dir = os.path.join('results', 'figures')", "out_dir = session.figures_dir")
    content = content.replace("out_dir = os.path.join('results', 'tables')", "out_dir = session.tables_dir")
    content = content.replace("save_path = os.path.join('results', 'figures', 'week3_task_graph.png')", "save_path = os.path.join(session.figures_dir, 'week3_task_graph.png')")
    content = content.replace("save_path = os.path.join('results', 'figures', 'week4_partition_output.png')", "save_path = os.path.join(session.figures_dir, 'week4_partition_output.png')")
    
    # Also fix plotters.py which pulls from hardcoded tables paths
    if 'plotters.py' in file_path:
        content = content.replace("csv_path = 'results/tables/table_III_sensitivity.csv'", "csv_path = os.path.join(session.tables_dir, 'table_III_sensitivity.csv')")
        content = content.replace("csv_path = 'results/tables/table_IV_tradeoffs.csv'", "csv_path = os.path.join(session.tables_dir, 'table_IV_tradeoffs.csv')")
        content = content.replace("self.output_dir = output_dir", "self.output_dir = session.figures_dir\n        session.setup_directories()")
        content = content.replace("existing = glob.glob(self.output_dir + '/week7_*.png')", "existing = glob.glob(session.figures_dir + '/week7_*.png')")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Patching complete!")
