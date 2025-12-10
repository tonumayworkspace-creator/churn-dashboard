import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "cleaned_churn_data.csv"

def main():
    df = pd.read_csv(CSV_FILE)

    # Show first few rows
    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    # Plot churn distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution")
    plt.savefig("plot_churn_distribution.png")
    print("\nSaved: plot_churn_distribution.png")

    # Plot tenure distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["tenure"], kde=True)
    plt.title("Tenure Distribution")
    plt.savefig("plot_tenure_distribution.png")
    print("Saved: plot_tenure_distribution.png")

    # Correlation heatmap (numerical only)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("plot_correlation_heatmap.png")
    print("Saved: plot_correlation_heatmap.png")

if __name__ == "__main__":
    main()
