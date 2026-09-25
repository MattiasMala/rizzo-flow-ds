import sys
from pathlib import Path

# The package lives in deadcells/deadcells; tests run from anywhere without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
