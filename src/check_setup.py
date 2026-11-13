
import importlib
import sys

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    create_directories,
)

PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "sqlalchemy",
    "streamlit",
    "plotly",
    "joblib",
    "pytest",
    "jupyterlab",
]


def check_python():
    print("\nChecking Python version...")

    version = sys.version_info

    print(
        f"Python {version.major}."
        f"{version.minor}.{version.micro}"
    )

    if version.major == 3 and version.minor == 13:
        print("[OK] Python 3.13 detected")
        return True

    print("[WARNING] Project targets Python 3.13")
    return False


def check_packages():
    print("\nChecking installed libraries...")

    missing = []

    for package in PACKAGES:
        try:
            importlib.import_module(package)
            print(f"[OK] {package}")

        except ImportError:
            print(f"[MISSING] {package}")
            missing.append(package)

    return missing


def check_directories():
    print("\nChecking directories...")

    create_directories()

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]

    success = True

    for directory in directories:
        if directory.is_dir():
            print(f"[OK] {directory}")
        else:
            print(f"[FAILED] {directory}")
            success = False

    return success


def main():
    print("=" * 55)
    print("CUSTOMER CHURN ANALYTICS")
    print("PHASE 1 - ENVIRONMENT VERIFICATION")
    print("=" * 55)

    print(f"\nProject: {BASE_DIR}")

    python_ok = check_python()
    missing = check_packages()
    folders_ok = check_directories()

    print("\n" + "=" * 55)

    if python_ok and not missing and folders_ok:
        print("SUCCESS: Environment is ready!")
        return 0

    print("FAILED: Please fix the issues above.")

    if missing:
        print("Missing packages:", ", ".join(missing))

    return 1


if __name__ == "__main__":
    sys.exit(main())
