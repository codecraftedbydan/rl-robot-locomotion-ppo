import sys
from pathlib import Path

# The training scripts import their siblings directly (e.g. ``from gae import ...``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
