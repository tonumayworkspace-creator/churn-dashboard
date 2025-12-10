import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_FILE = "cleaned_churn_data.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    print("Original shape:", df.shape)

    # Convert target column to binary (Yes=1, No=0)
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    print("After encoding:", X.shape)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save processed datasets
    pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

    print("Preprocessing completed.")
    print("Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

if __name__ == "__main__":
    main()
