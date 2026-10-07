import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from src.predict import predict_session

st.set_page_config(page_title="Cart Abandonment Predictor", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
        }
        div[class*="st-key-results_scroll"] div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap;
            overflow-x: auto;
            padding-bottom: 1rem;
        }
        div[class*="st-key-results_scroll"] div[data-testid="column"] {
            min-width: 420px;
            flex: 0 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Cart Abandonment Predictor")
st.caption("Predicts the probability that an e-commerce session ends without a purchase, "
           "explains why, and suggests a business action. Trained on the UCI Online Shoppers "
           "Purchasing Intention dataset.")

# --------------------------------------------------
# Session state init
# --------------------------------------------------
if "preset" not in st.session_state:
    st.session_state.preset = None
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# Sidebar — risk thresholds
# --------------------------------------------------
with st.sidebar:
    st.header("Risk thresholds")
    low = st.slider("Medium risk starts at", 0.0, 1.0, 0.80, 0.01)
    high = st.slider("High risk starts at", 0.0, 1.0, 0.98, 0.01)
    st.caption("Defaults were tuned for this dataset, where about 84% of sessions end without a purchase.")
    if low >= high:
        st.error("The Medium threshold must be lower than the High threshold.")
        st.stop()

    st.divider()
    if st.session_state.history:
        st.caption(f"{len(st.session_state.history)} prediction(s) this session.")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# --------------------------------------------------
# Quick presets
# --------------------------------------------------
st.write("**Quick presets** (for demo purposes):")
preset_col1, preset_col2, preset_col3 = st.columns(3)
if preset_col1.button("🔴 High-risk session"):
    st.session_state.preset = "high"
if preset_col2.button("🟢 Low-risk session"):
    st.session_state.preset = "low"
if preset_col3.button("↺ Reset to defaults"):
    st.session_state.preset = None

defaults = {
    "high": dict(pp=1, ps=0.0, pv=0.0, br=0.2, er=0.2, month="May", vt="New_Visitor"),
    "low":  dict(pp=8, ps=300.0, pv=30.0, br=0.02, er=0.05, month="Nov", vt="New_Visitor"),
}.get(st.session_state.preset, dict(pp=8, ps=300.0, pv=0.0, br=0.02, er=0.05, month="Nov", vt="New_Visitor"))

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["New_Visitor", "Returning_Visitor", "Other"]

# --------------------------------------------------
# Form — tabbed layout
# --------------------------------------------------
with st.form("session_form"):
    tab1, tab2, tab3 = st.tabs(["Customer & Session", "Browsing Behaviour", "Technical"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        customer_id = c1.text_input("Customer ID", "C001")
        session_id = c2.text_input("Session ID", "S1001")
        visitor = c3.selectbox("Visitor type", VISITOR_TYPES, index=VISITOR_TYPES.index(defaults["vt"]))

        c1, c2, c3 = st.columns(3)
        month = c1.selectbox("Month", MONTHS, index=MONTHS.index(defaults["month"]))
        weekend = c2.checkbox("Weekend session")
        special_day = c3.select_slider("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], value=0.0)

    with tab2:
        c1, c2, c3 = st.columns(3)
        product_pages = c1.number_input("Product pages viewed", 0, 800, defaults["pp"])
        product_secs = c2.number_input("Time on product pages (sec)", 0.0, 70000.0, defaults["ps"])
        page_values = c3.number_input(
            "Page value", 0.0, 400.0, defaults["pv"],
            help="Analytics-derived metric: average value of the pages visited before a transaction. "
                 "Use 0 if unknown. The model relies on it heavily.")

        c1, c2, c3, c4 = st.columns(4)
        admin_pages = c1.number_input("Account/admin pages", 0, 30, 0)
        admin_secs = c2.number_input("Admin time (sec)", 0.0, 4000.0, 0.0)
        info_pages = c3.number_input("Info pages", 0, 30, 0)
        info_secs = c4.number_input("Info time (sec)", 0.0, 3000.0, 0.0)

        c1, c2 = st.columns(2)
        bounce = c1.slider("Bounce rate", 0.0, 0.2, defaults["br"], 0.005)
        exit_rate = c2.slider("Exit rate", 0.0, 0.2, defaults["er"], 0.005)

    with tab3:
        c1, c2, c3, c4 = st.columns(4)
        os_ = c1.number_input("Operating system (code)", 1, 8, 2)
        browser = c2.number_input("Browser (code)", 1, 13, 2)
        region = c3.number_input("Region (code)", 1, 9, 1)
        traffic = c4.number_input("Traffic source (code)", 1, 20, 2)

    submitted = st.form_submit_button("Predict", type="primary")

# --------------------------------------------------
# Prediction + results
# --------------------------------------------------
if submitted:
    raw = {
        "Administrative": admin_pages,
        "Administrative_Duration": admin_secs,
        "Informational": info_pages,
        "Informational_Duration": info_secs,
        "ProductRelated": product_pages,
        "ProductRelated_Duration": product_secs,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "OperatingSystems": os_,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "Weekend": weekend,
        "Month": month,
        "VisitorType": visitor,
    }

    result = predict_session(raw, customer_id, session_id, low=low, high=high)
    st.session_state.history.append({
        "Customer": customer_id,
        "Session": session_id,
        "Abandonment %": f"{result['abandonment_probability']:.1%}",
        "Risk": result["risk_level"],
        "Action": result["recommended_action"],
    })

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Abandonment probability", f"{result['abandonment_probability']:.1%}")
    m2.metric("Risk level", result["risk_level"])
    m3.metric("Recommended action", result["recommended_action"])

    banner = {"High": st.error, "Medium": st.warning, "Low": st.success}[result["risk_level"]]
    banner(f"{result['risk_level']} risk: {result['recommended_action']}")

    # ---- Results (scrollable row) ----
    st.divider()
    st.subheader("Results")

    with st.container(key="results_scroll"):
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.markdown("**Abandonment risk**")
            gauge_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}[result["risk_level"]]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["abandonment_probability"] * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, low * 100], "color": "#d4f4dd"},
                        {"range": [low * 100, high * 100], "color": "#fde3b8"},
                        {"range": [high * 100, 100], "color": "#f8d0cc"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3},
                                  "thickness": 0.8, "value": result["abandonment_probability"] * 100},
                },
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_gauge, width='stretch')

        with rc2:
            st.markdown("**Important factors**")
            if result["important_factors"]:
                fdf = pd.DataFrame(result["important_factors"])
                fdf["signed_impact"] = fdf.apply(
                    lambda r: r["impact"] if r["effect"] == "raises abandonment risk" else -r["impact"], axis=1)
                fdf = fdf.sort_values("signed_impact")

                fig_bar = go.Figure(go.Bar(
                    x=fdf["signed_impact"], y=fdf["factor"], orientation="h",
                    marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in fdf["signed_impact"]],
                    text=[f"{v:+.2f}" for v in fdf["signed_impact"]], textposition="outside",
                ))
                fig_bar.update_layout(
                    xaxis_title="← lowers risk    raises risk →",
                    height=300, margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_bar, width='stretch')
            else:
                st.write("No single factor had a large effect.")

        with rc3:
            st.markdown("**What-if: Page Value**")
            pv_range = np.linspace(0, 60, 25)
            probs = []
            for pv in pv_range:
                test = {**raw, "PageValues": pv}
                probs.append(predict_session(test, low=low, high=high)["abandonment_probability"])

            fig_line = go.Figure(go.Scatter(
                x=pv_range, y=[p * 100 for p in probs], mode="lines+markers",
                line=dict(color="#3498db", width=3),
            ))
            fig_line.add_vline(x=page_values, line_dash="dash", line_color="gray",
                                annotation_text="this session")
            fig_line.update_layout(
                xaxis_title="Page Value", yaxis_title="Abandonment Probability (%)",
                height=300, margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_line, width='stretch')

    with st.expander("API-style JSON output"):
        st.code(json.dumps(result, indent=2), language="json")

# --------------------------------------------------
# Prediction history
# --------------------------------------------------
if st.session_state.history:
    st.divider()
    st.subheader("Prediction history (this session)")
    st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True, width='stretch')