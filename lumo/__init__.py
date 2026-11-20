"""Lumo — a from-scratch AI agent with a web UI. No provider APIs anywhere."""
import os

# Keep the sandbox responsive while training (must run before numpy imports).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

__version__ = "0.1.0"
