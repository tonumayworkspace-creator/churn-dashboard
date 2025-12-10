import pandas as pd

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    print("\n=== BEFORE CLEANING ===")
    print(df.dtypes)

    # Fix the TotalCharges column
    # Convert to numeric and force errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Check how many became NaN
    missing_total = df["TotalCharges"].isna().sum()
    print("\nMissing TotalCharges values after conversion:", missing_total)

    # Drop rows where TotalCharges is NaN (only a few rows)
    df = df.dropna(subset=["TotalCharges"])

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    print("\n=== AFTER CLEANING ===")
    print(df.dtypes)

    # Save cleaned dataset for next steps
    df.to_csv("cleaned_churn_data.csv", index=False)
    print("\nCleaned dataset saved as cleaned_churn_data.csv")

if __name__ == "__main__":
    main()
