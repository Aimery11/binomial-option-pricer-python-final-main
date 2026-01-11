import os
import subprocess
import venv
import shutil
import platform
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / ".venv"
IS_WIN = platform.system() == "Windows"


def py_exec_in_venv() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python")


def run(cmd, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def recreate_venv():
    if VENV_DIR.is_dir():
        print(">> Removing previous .venv (start clean)…")
        shutil.rmtree(VENV_DIR, ignore_errors=True)
    print(">> Creating isolated environment (.venv)…")
    venv.create(VENV_DIR, with_pip=True)


def ensure_venv_exists():
    if not VENV_DIR.is_dir():
        print(">> No .venv found → creating it…")
        venv.create(VENV_DIR, with_pip=True)


def install_deps(py_exe: Path):
    print(">> Upgrading pip…")
    run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    req = HERE / "requirements.txt"
    if req.is_file():
        print(">> Installing requirements.txt…")
        try:
            run([str(py_exe), "-m", "pip", "install", "-r", str(req), "--upgrade"], check=True)
            return
        except subprocess.CalledProcessError:
            print("!! requirements.txt install failed, falling back to safe pins…")

    print(">> Installing safe fallback pins…")
    pins = [
        "numpy==2.1.3",
        "pandas==2.2.2",
        "yfinance==0.2.66",
        "matplotlib==3.9.2",
        "mplcursors==0.5.3",
        "ipympl==0.9.5",
        "requests==2.32.3",
    ]
    run([str(py_exe), "-m", "pip", "install", *pins], check=True)


def macos_unquarantine(py_exe: Path):
    if platform.system() != "Darwin":
        return
    try:
        site = subprocess.check_output(
            [str(py_exe), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True
        ).strip()
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", site], check=False)
    except Exception:
        pass


def launch_app(py_exe: Path):
    app = HERE / "internal_skeleton.py"
    if not app.is_file():
        print("!! ERROR: internal_skeleton.py not found in the project folder.")
        raise SystemExit(2)

    print("\n>> Launching pricer…\n")
    proc = run([str(py_exe), str(app)], check=False)
    raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Option Pricer in an isolated virtual environment (.venv)."
    )
    parser.add_argument("--clean", action="store_true", help="Recreate the .venv from scratch.")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation.")
    args = parser.parse_args()

    os.chdir(HERE)

    try:
        if args.clean:
            recreate_venv()
        else:
            ensure_venv_exists()

        py_exe = py_exec_in_venv()

        # If the venv python isn't there, venv creation failed (permissions, etc.)
        if not py_exe.is_file():
            print("!! ERROR: venv python not found. venv creation likely failed.")
            raise SystemExit(2)

        if not args.skip_install:
            install_deps(py_exe)

        macos_unquarantine(py_exe)

        # Quick sanity check (helps recruiters debug instantly)
        run([str(py_exe), "-c", "import yfinance as yf; print('yfinance:', yf.__version__)"], check=False)

        launch_app(py_exe)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except subprocess.CalledProcessError as e:
        print("\n!! A command failed:")
        print("   ", " ".join(e.cmd) if isinstance(e.cmd, list) else e.cmd)
        raise SystemExit(e.returncode)
    except Exception as e:
        print("\n!! Unexpected error:", str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()