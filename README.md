# Churn Dashboard – Telco Customer Churn

End-to-end data science project:  
• Telco customer churn analysis  
• Logistic Regression model  
• Flask web dashboard with SHAP explainability, login/signup, and modern UI  

## 1. Project structure

- `01_load_data.py` – load raw Telco churn data  
- `02_data_overview.py` – data overview & sanity checks  
- `03_data_cleaning.py` – cleaning, type fixes, missing values  
- `04_basic_eda.py` – basic exploratory data analysis & plots  
- `05_preprocessing.py` – encoding, train/test split, scaling  
- `06_modeling.py` – model training & evaluation  
- `07_shap_explainability.py` – SHAP-based feature importance  
- `app_flask.py` – Flask app for the churn prediction dashboard  
- `templates/` – HTML templates (dashboard + result pages)  
- `static/` – CSS, JavaScript, and SHAP images  

## 2. How to run locally

```bash
# 1) Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the Flask app
python app_flask.py
