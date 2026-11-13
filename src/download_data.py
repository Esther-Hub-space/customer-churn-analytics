
from io import BytesIO
from urllib.request import urlopen
import pandas as pd

from src.config import RAW_DATA_FILE, create_directories

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

REQUIRED_COLUMNS = {
    "customerID",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def download_dataset():
    create_directories()

    print("Downloading IBM Telco Customer Churn dataset...")

    try:
        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()

        df = pd.read_csv(BytesIO(content))

        missing = REQUIRED_COLUMNS - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing expected columns: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError("Downloaded dataset is empty.")

        RAW_DATA_FILE.write_bytes(content)

        print("Download completed successfully!")
        print(f"Saved to: {RAW_DATA_FILE}")
        print(f"Customers: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

    except Exception as error:
        print(f"Download failed: {error}")
        raise


if __name__ == "__main__":
    download_dataset()
