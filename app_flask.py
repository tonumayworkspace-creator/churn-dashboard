# app_flask.py (REPLACE existing file)
import os
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from functools import wraps

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    g,
)

from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# ---------- CONFIG ----------
# You should set a stronger secret in production (e.g., env var)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_to_a_secure_random_value")

# Paths & constants
MODEL_PATH = Path("models") / "logistic_model.joblib"
FEATURES_PATH = Path("X_train.csv")
LOG_PATH = Path("predictions_log.csv")
STATIC_DIR = Path("static")
SHAP_DIR = STATIC_DIR / "shap"
DB_PATH = Path("users.db")
os.makedirs(SHAP_DIR, exist_ok=True)

# ---------- DB (SQLite) helpers ----------
def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def create_user(username, email, password_plain):
    db = get_db()
    pw_hash = generate_password_hash(password_plain)
    now = datetime.utcnow().isoformat()
    try:
        db.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, pw_hash, now),
        )
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def find_user_by_username(username):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return row

def verify_user(username, password_plain):
    row = find_user_by_username(username)
    if row:
        return check_password_hash(row["password_hash"], password_plain)
    return False

# ---------- existing ML app setup ----------
# sanity checks
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run modeling step first.")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}. Run preprocessing step first.")

# load model & features
model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(FEATURES_PATH)
FEATURE_COLUMNS = X_train.columns.tolist()

# attempt to compute simple model metrics if test set exists
MODEL_METRICS = {"accuracy": None, "roc_auc": None, "test_size": 0}
try:
    if Path("X_test.csv").exists() and Path("y_test.csv").exists():
        X_test = pd.read_csv("X_test.csv")
        y_test = pd.read_csv("y_test.csv")["Churn"]
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score, roc_auc_score

        MODEL_METRICS["accuracy"] = float(accuracy_score(y_test, preds))
        MODEL_METRICS["roc_auc"] = float(roc_auc_score(y_test, proba))
        MODEL_METRICS["test_size"] = int(len(y_test))
except Exception as e:
    print("Warning computing model metrics:", e)

# SHAP explainer initialization (done once)
try:
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    print("SHAP explainer initialized.")
except Exception as e:
    explainer = None
    print("Warning: Could not create SHAP explainer:", e)

# ---------- helper: login required decorator ----------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function

# ---------- helper: prepare input (unchanged plus new fields) ----------
def prepare_input(form_data):
    tenure = float(form_data.get("tenure", 0))
    monthly_charges = float(form_data.get("monthly_charges", 0))
    total_charges = float(form_data.get("total_charges", 0))

    contract = form_data.get("contract", "")
    internet_service = form_data.get("internet_service", "")
    payment_method = form_data.get("payment_method", "")

    # NEW fields we added earlier
    online_security = form_data.get("OnlineSecurity", "No")
    tech_support = form_data.get("TechSupport", "No")

    input_base = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "InternetService": [internet_service],
        "PaymentMethod": [payment_method],
        "OnlineSecurity": [online_security],
        "TechSupport": [tech_support]
    })

    input_encoded = pd.get_dummies(input_base, drop_first=False)
    all_features = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    for col in input_encoded.columns:
        if col in all_features.columns:
            all_features[col] = input_encoded[col].values

    # NOTE: load and apply scaler here if you saved one (recommended)
    return all_features

# ---------- helper: shap plot & logging (unchanged) ----------
def create_shap_bar(image_name_prefix: str, feature_names, shap_values_for_sample, k=5):
    arr = np.array(shap_values_for_sample).flatten()
    abs_idx = np.argsort(np.abs(arr))[::-1][:k]
    top_idx = abs_idx[::-1]
    top_feats = [feature_names[i] for i in top_idx]
    top_vals = arr[top_idx]

    plt.figure(figsize=(6, 3))
    y_pos = np.arange(len(top_feats))
    colors = ["#ef4444" if v > 0 else "#10b981" for v in top_vals]
    plt.barh(y_pos, top_vals, color=colors)
    plt.yticks(y_pos, top_feats, fontsize=9)
    plt.xlabel("SHAP value (feature contribution)")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()

    timestamp = int(time.time() * 1000)
    fname = f"{image_name_prefix}_{timestamp}.png"
    save_path = SHAP_DIR / fname
    plt.savefig(save_path, dpi=150)
    plt.close()

    return f"shap/{fname}", list(zip(top_feats, top_vals.tolist()))

def append_prediction_log(timestamp_iso, probability, prediction, top_features_text):
    header_needed = not LOG_PATH.exists()
    df_row = pd.DataFrame([{
        "timestamp": timestamp_iso,
        "probability": probability,
        "prediction": int(prediction),
        "top_features": top_features_text
    }])
    if header_needed:
        df_row.to_csv(LOG_PATH, index=False, mode="w")
    else:
        df_row.to_csv(LOG_PATH, index=False, mode="a", header=False)

def read_last_predictions(n=5):
    if not LOG_PATH.exists():
        return []
    df = pd.read_csv(LOG_PATH)
    df = df.sort_values("timestamp", ascending=False).head(n)
    results = df.to_dict(orient="records")
    return results

# ---------- FLASK routes: Auth ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    init_db()  # ensure DB exists
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("signup.html")
        ok = create_user(username, email, password)
        if not ok:
            flash("Username or email already exists. Choose another.", "danger")
            return render_template("signup.html")
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    init_db()
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_user(username, password):
            session["user"] = username
            flash(f"Welcome, {username}!", "success")
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        flash("Invalid username or password.", "danger")
        return render_template("login.html")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("login"))

# ---------- FLASK routes: ML app ----------
@app.route("/", methods=["GET"])
@login_required
def index():
    contracts = ["Month-to-month", "One year", "Two year"]
    internet_services = ["DSL", "Fiber optic", "No"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    # choices for new fields
    online_security_choices = ["No", "Yes", "No internet service"]
    tech_support_choices = ["No", "Yes", "No internet service"]
    user = session.get("user")
    return render_template(
        "index.html",
        contracts=contracts,
        internet_services=internet_services,
        payment_methods=payment_methods,
        online_security_choices=online_security_choices,
        tech_support_choices=tech_support_choices,
        user=user
    )

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form = request.form
    all_features = prepare_input(form)

    proba = model.predict_proba(all_features)[0][1]
    pred = model.predict(all_features)[0]

    shap_image_rel = None
    top_feats_info = []

    try:
        if explainer is not None:
            shap_vals = explainer.shap_values(all_features)
            shap_image_rel, top_feats_info = create_shap_bar("shap", FEATURE_COLUMNS, shap_vals, k=5)
    except Exception as e:
        print("Warning computing SHAP for request:", e)

    timestamp_iso = datetime.utcnow().isoformat()
    top_text = ", ".join([f"{name}:{val:.3f}" for name, val in top_feats_info[:5]]) if top_feats_info else ""
    append_prediction_log(timestamp_iso, float(proba), int(pred), top_text)

    last_preds = read_last_predictions(5)
    metrics = {
        "accuracy": MODEL_METRICS.get("accuracy"),
        "roc_auc": MODEL_METRICS.get("roc_auc"),
        "test_size": MODEL_METRICS.get("test_size"),
    }

    result = {
        "prediction": int(pred),
        "probability": float(proba),
        "probability_pct": round(float(proba) * 100, 2),
        "shap_image": shap_image_rel,
    }

    return render_template("result.html", result=result, metrics=metrics, last_predictions=last_preds, user=session.get("user"))

if __name__ == "__main__":
    # ensure DB exists
    with app.app_context():
        init_db()
    app.run(host="0.0.0.0", port=8501, debug=True)
