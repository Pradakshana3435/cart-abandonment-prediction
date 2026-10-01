# Cart Abandonment Prediction Model

An AI/ML proof-of-concept that predicts the probability of an e-commerce
customer abandoning their session without completing a purchase, and
recommends a business action based on the predicted risk level.

## What it does
Given session-level browsing data (pages viewed, time spent, bounce/exit
rates, visitor type, etc.), the model outputs:
- Abandonment probability
- Risk level (Low / Medium / High, with configurable thresholds)
- Key factors driving the prediction (via SHAP)
- A recommended business action (reminder, discount offer, etc.)

## Dataset
Built on the UCI "Online Shoppers Purchasing Intention" dataset
(12,330 sessions). Note: this dataset has no cart-specific fields
(cart value, checkout status), so `Revenue = False` was used as a proxy
for session abandonment, and cart-value/checkout signals were
approximated from available behavioral features. This trade-off is
documented in the full report.

## Approach
- EDA + class imbalance analysis (~85% no-purchase / 15% purchase)
- Compared Logistic Regression, Random Forest, and XGBoost
  (5-fold CV, ROC-AUC / PR-AUC)
- Selected XGBoost (PR-AUC 0.75) for best ranking performance and
  native SHAP support
- Calibration check: dropped class-weighting after confirming it
  distorted probability estimates without improving ranking
- Risk thresholds tuned against the dataset's actual base rate,
  validated on out-of-fold predictions to avoid test-set leakage

## Demo
Interactive Streamlit app (`app/app.py`) — input session details,
get a live prediction with explanation and recommended action.

## Tech stack
Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit

## Project status
Proof-of-concept / academic project. See `/notebooks` for the full
EDA, modelling, and evaluation walkthrough.

## How to run it locally

**1. Clone the repo and enter the folder**
```cmd
git clone https://github.com/yourusername/cart-abandonment-prediction.git
cd cart-abandonment-prediction
```

**2. Create and activate a virtual environment**
```cmd
py -3.13 -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```cmd
pip install -r requirements.txt
```

**4. Run the notebooks (optional — to see the full EDA/modelling process)**
```cmd
jupyter notebook
```
Open `notebooks/01_eda.ipynb` and run cells in order. This also regenerates
the saved model at `models/cart_abandonment_xgb.joblib` if it isn't
already present.

**5. Run the Streamlit demo**
```cmd
streamlit run app\app.py
```
Opens in your browser at `http://localhost:8501`. Fill in the session
details form and click **Predict** to see the abandonment probability,
risk level, important factors, and recommended action.

## Requirements
- Python 3.13 (Note: on Windows, Python 3.14 may trigger Smart App
  Control blocks on compiled scikit-learn/scipy files — 3.13 is
  recommended)