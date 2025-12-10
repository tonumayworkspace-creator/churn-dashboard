import pandas as pd

# File name of the dataset
CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def main():
    # Load the dataset
    df = pd.read_csv(CSV_FILE)

    # Print basic information
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

if __name__ == "__main__":
    main()
