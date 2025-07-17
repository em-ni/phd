import sys
import subprocess

# Print the Python version
print(sys.version)

# List installed packages
installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
print(installed_packages)
