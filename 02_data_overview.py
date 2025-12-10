import pandas as pd

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    print("\n=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isnull().sum())

    print("\n=== UNIQUE VALUES (for categorical understanding) ===")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].nunique()} unique values")

if __name__ == "__main__":
    main()
