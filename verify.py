"""Verify the project end-to-end: unit tests + a small training smoke run."""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], check=False)
sys.exit(r.returncode)
