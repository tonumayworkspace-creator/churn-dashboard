# 06_modeling.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import joblib

# File names produced earlier
X_TRAIN = "X_train.csv"
X_TEST = "X_test.csv"
Y_TRAIN = "y_train.csv"
Y_TEST = "y_test.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "logistic_model.joblib")

def main():
    # --- load data ---
    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN)["Churn"]
    y_test = pd.read_csv(Y_TEST)["Churn"]

    print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # --- train model ---
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained.")

    # --- predictions ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability for positive class

    # --- evaluation metrics ---
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC AUC : {roc_auc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # --- confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # --- save model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model to: {MODEL_FILE}")

    # --- plots ---
    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved: confusion_matrix.png")

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    print("Saved: roc_curve.png")

if __name__ == "__main__":
    main()
