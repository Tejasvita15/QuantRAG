"""Create the project environment without replacing local credentials.

Run with Python 3.10 or 3.11: python scripts/setup.py
"""

import argparse
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Create environment/config only")
    args = parser.parse_args()
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        parser.error("Use Python 3.10 or 3.11, e.g. py -3.10 scripts/setup.py on Windows.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print("Creating .venv", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] in {(3,10),(3,11)}, 'Existing .venv requires Python 3.10 or 3.11'"], check=True)
    if not args.skip_install:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "check"], check=True)
    config = ROOT / ".env"
    if not config.exists():
        shutil.copyfile(ROOT / ".env.example", config)
        print("Created .env from .env.example; add your own credentials locally.")
    else:
        print("Existing .env preserved.")
    print("Setup complete. Run scripts/start.ps1 (Windows) or bash scripts/start.sh (macOS/Linux).")
    print("Route only works without data or credentials. See docs/team_setup.md for SEC indexes.")


if __name__ == "__main__":
    main()
