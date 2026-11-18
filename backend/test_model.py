"""Forwarder for test_model.py located in backend/scripts/."""
import subprocess
import sys
from pathlib import Path

script = Path(__file__).parent / "scripts" / "test_model.py"
sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
