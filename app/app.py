import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
import streamlit as st
from src.predict import predict_session

st.set_page_config(page_title="Cart Abandonment Predictor", layout="wide")
st.title("Cart Abandonment Predictor")
st.caption("Predicts the probability that an e-commerce session ends without a purchase, "
           "explains why, and suggests a business action. Trained on the UCI Online Shoppers "
           "Purchasing Intention dataset.")

with st.sidebar:
    st.header("Risk thresholds")
    low = st.slider("Medium risk starts at", 0.0, 1.0, 0.80, 0.01)
    high = st.slider("High risk starts at", 0.0, 1.0, 0.98, 0.01)
    st.caption("Defaults were tuned for this dataset, where about 84% of sessions end without a purchase.")
    if low >= high:
        st.error("The Medium threshold must be lower than the High threshold.")
        st.stop()

with st.form("session_form"):
    st.subheader("Customer and session")
    c1, c2, c3 = st.columns(3)
    customer_id = c1.text_input("Customer ID", "C001")
    session_id = c2.text_input("Session ID", "S1001")
    visitor = c3.selectbox("Visitor type", ["New_Visitor", "Returning_Visitor", "Other"])

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
    weekend = c2.checkbox("Weekend session")
    special_day = c3.select_slider("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

    st.subheader("Browsing behaviour")
    c1, c2, c3 = st.columns(3)
    product_pages = c1.number_input("Product pages viewed", 0, 800, 8)
    product_secs = c2.number_input("Time on product pages (sec)", 0.0, 70000.0, 300.0)
    page_values = c3.number_input(
        "Page value", 0.0, 400.0, 0.0,
        help="Analytics-derived metric: average value of the pages visited before a transaction. "
             "Use 0 if unknown. The model relies on it heavily.")

    c1, c2, c3, c4 = st.columns(4)
    admin_pages = c1.number_input("Account/admin pages", 0, 30, 0)
    admin_secs = c2.number_input("Admin time (sec)", 0.0, 4000.0, 0.0)
    info_pages = c3.number_input("Info pages", 0, 30, 0)
    info_secs = c4.number_input("Info time (sec)", 0.0, 3000.0, 0.0)

    c1, c2 = st.columns(2)
    bounce = c1.slider("Bounce rate", 0.0, 0.2, 0.02, 0.005)
    exit_rate = c2.slider("Exit rate", 0.0, 0.2, 0.05, 0.005)

    st.subheader("Technical")
    c1, c2, c3, c4 = st.columns(4)
    os_ = c1.number_input("Operating system (code)", 1, 8, 2)
    browser = c2.number_input("Browser (code)", 1, 13, 2)
    region = c3.number_input("Region (code)", 1, 9, 1)
    traffic = c4.number_input("Traffic source (code)", 1, 20, 2)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    raw = {
        "Administrative": admin_pages, "Administrative_Duration": admin_secs,
        "Informational": info_pages, "Informational_Duration": info_secs,
        "ProductRelated": product_pages, "ProductRelated_Duration": product_secs,
        "BounceRates": bounce, "ExitRates": exit_rate, "PageValues": page_values,
        "SpecialDay": special_day, "OperatingSystems": os_, "Browser": browser,
        "Region": region, "TrafficType": traffic, "Weekend": weekend,
        "Month": month, "VisitorType": visitor,
    }
    result = predict_session(raw, customer_id, session_id, low=low, high=high)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Abandonment probability", f"{result['abandonment_probability']:.1%}")
    m2.metric("Risk level", result["risk_level"])
    m3.metric("Recommended action", result["recommended_action"])

    banner = {"High": st.error, "Medium": st.warning, "Low": st.success}[result["risk_level"]]
    banner(f"{result['risk_level']} risk: {result['recommended_action']}")

    st.subheader("Important factors")
    if result["important_factors"]:
        st.dataframe(pd.DataFrame(result["important_factors"]), hide_index=True, use_container_width=True)
    else:
        st.write("No single factor had a large effect.")

    with st.expander("API-style JSON output"):
        st.code(json.dumps(result, indent=2), language="json")