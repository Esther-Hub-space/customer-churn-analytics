
from pathlib import Path

# Main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Other project directories
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Dataset filenames
RAW_DATA_FILE = RAW_DATA_DIR / "customer_churn.csv"
CLEAN_DATA_FILE = (
    PROCESSED_DATA_DIR / "cleaned_churn.csv"
)

# Machine learning configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20


def create_directories():
    """Create project directories if missing."""

    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        NOTEBOOKS_DIR,
        DASHBOARD_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":
    create_directories()

    print("Project directories created!")
    print(f"Project root: {BASE_DIR}")
