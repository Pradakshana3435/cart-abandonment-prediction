from pathlib import Path
import joblib
import pandas as pd
import shap

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cart_abandonment_xgb.joblib"

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_columns = _bundle["columns"]
_explainer = shap.TreeExplainer(_model)

READABLE = {
    "PageValues": "Page value",
    "ExitRates": "Exit rate",
    "BounceRates": "Bounce rate",
    "ProductRelated": "Product pages viewed",
    "ProductRelated_Duration": "Time on product pages",
    "Administrative": "Account/admin pages viewed",
    "Administrative_Duration": "Time on account/admin pages",
    "Informational": "Info pages viewed",
    "Informational_Duration": "Time on info pages",
    "TrafficType": "Traffic source",
}


def risk_level(p, low=None, high=None):
    low = _bundle["low"] if low is None else low
    high = _bundle["high"] if high is None else high
    if p >= high:
        return "High"
    if p >= low:
        return "Medium"
    return "Low"


def recommend_action(risk, is_new, high_value, checkout_proxy):
    if risk == "Low":
        return "No immediate action"
    if risk == "Medium":
        return "Cart reminder"
    if is_new:
        return "First-purchase incentive"
    if high_value:
        return "Discount / free shipping offer"
    if checkout_proxy:
        return "Checkout reminder"
    return "Cart reminder"


def top_factors(row, k=5, min_impact=0.05):
    contrib = pd.Series(-_explainer(row).values[0], index=row.columns)  # + = raises abandonment
    factors = []
    for feat in contrib.abs().sort_values(ascending=False).index:
        val = float(row.iloc[0][feat])
        if feat.startswith(("Month_", "VisitorType_")) and val == 0:
            continue
        if abs(contrib[feat]) < min_impact:
            break
        if feat.startswith("Month_"):
            label = f"Month: {feat[6:]}"
        elif feat.startswith("VisitorType_"):
            label = f"Visitor type: {feat[12:]}"
        else:
            label = READABLE.get(feat, feat)
        factors.append({
            "factor": label,
            "value": val,
            "effect": "raises abandonment risk" if contrib[feat] > 0 else "lowers abandonment risk",
            "impact": round(float(abs(contrib[feat])), 3),
        })
        if len(factors) == k:
            break
    return factors


def predict_session(raw, customer_id=None, session_id=None, low=None, high=None):
    row = pd.DataFrame(0.0, index=[0], columns=_columns)
    for k, v in raw.items():
        if k in ("Month", "VisitorType"):
            col = f"{k}_{v}"
            if col in row.columns:  # Aug and New_Visitor are the dropped baselines
                row.loc[0, col] = 1.0
        elif k in row.columns:
            row.loc[0, k] = float(v)

    p_ab = float(1 - _model.predict_proba(row)[0, 1])
    risk = risk_level(p_ab, low, high)

    is_new = row.loc[0, "VisitorType_Returning_Visitor"] == 0 and row.loc[0, "VisitorType_Other"] == 0
    high_value = row.loc[0, "ProductRelated"] >= _bundle["hv_cut"]
    checkout_proxy = row.loc[0, "PageValues"] > 0

    return {
        "customer_id": customer_id,
        "session_id": session_id,
        "abandonment_probability": round(p_ab, 3),
        "risk_level": risk,
        "recommended_action": recommend_action(risk, is_new, high_value, checkout_proxy),
        "important_factors": top_factors(row),
    }