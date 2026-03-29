import os
import sys
from datetime import datetime

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SessionContext:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionContext, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
        
    def initialize(self):
        # Generate the timestamp for this particular interactive execution
        # Use env var to ensure subprocesses bind to the exact same session directory!
        ts = os.environ.get("ORCHESTRATOR_RUN_TS")
        if not ts:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.environ["ORCHESTRATOR_RUN_TS"] = ts
        self.timestamp = ts
        
        # Calculate base project directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define isolated save paths for this session
        self.run_dir = os.path.join(base_dir, 'results', 'runs', self.timestamp)
        self.figures_dir = os.path.join(self.run_dir, 'figures')
        self.tables_dir = os.path.join(self.run_dir, 'tables')
        
    def setup_directories(self):
        """Creates the timestamped directories physically on disk."""
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        return self.run_dir

    def log_info(self, message):
        print(f"{TerminalColors.OKCYAN}[INFO]{TerminalColors.ENDC} {message}")
        
    def log_success(self, message):
        print(f"{TerminalColors.OKGREEN}[SUCCESS]{TerminalColors.ENDC} {message}")
        
    def log_warning(self, message):
        print(f"{TerminalColors.WARNING}[WARNING]{TerminalColors.ENDC} {message}")
        
    def log_error(self, message):
        print(f"{TerminalColors.FAIL}[ERROR]{TerminalColors.ENDC} {message}")
        
    def log_header(self, message):
        print(f"\n{TerminalColors.HEADER}{TerminalColors.BOLD}{'=' * 60}")
        print(f"{message}")
        print(f"{'=' * 60}{TerminalColors.ENDC}")

# Expose a global singleton object
session = SessionContext()
