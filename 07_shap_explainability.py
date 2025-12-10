# 07_shap_explainability.py
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

X_TRAIN = "X_train.csv"
MODEL_FILE = "models/logistic_model.joblib"

def main():
    # Load the processed training data
    X_train = pd.read_csv(X_TRAIN)

    # Load the trained logistic regression model
    model = joblib.load(MODEL_FILE)

    print("Model and training data loaded.")

    # Initialize SHAP explainer
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_train)

    print("SHAP values calculated.")

    # --- Summary Plot (Global Feature Importance) ---
    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()
    print("Saved: shap_summary_plot.png")

    # --- Bar Plot (Feature Importance Bar Chart) ---
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png")
    plt.close()
    print("Saved: shap_feature_importance.png")

if __name__ == "__main__":
    main()
