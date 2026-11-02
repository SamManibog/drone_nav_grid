import subprocess
from pathlib import Path

subprocess.run([
    "pip",
    "install",
    "maturin",
])

subprocess.run([
    "maturin",
    "build",
    "--release",
    "--generate-stubs",
])

name_pattern = "drone_nav"
dir_path = Path(__file__).parent / "target" / "wheels"
matching_wheels = list(dir_path.glob(f"*{name_pattern}*.whl"))
if not matching_wheels:
    raise FileNotFoundError(f"No wheel files matching '{name_pattern}' found")

for wheel in matching_wheels:
    print(f"Attempting to install wheel '{str(wheel)}'.")

    result = subprocess.run([
        "pip",
        "install",
        str(wheel),
        "--force-reinstall",
        "--no-binary",
        ":all:",
        "--no-cache-dir",
    ])

    if result.returncode == 0:
        break;
