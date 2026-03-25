import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("loan_risk_model.joblib")

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Risk Predictor")
st.caption("XGBoost · AUC 0.988 · Accuracy 93%")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Borrower Profile")

col1, col2 = st.columns(2)

with col1:
    income           = st.number_input("Annual Income",        value=50000,  min_value=0, step=1000)
    loan_amount      = st.number_input("Loan Amount",          value=200000, min_value=0, step=5000)
    property_value   = st.number_input("Property Value",       value=300000, min_value=0, step=5000)

with col2:
    rate_of_interest = st.number_input("Rate of Interest (%)", value=8.5,    min_value=0.0, step=0.1, format="%.1f")
    upfront_charges  = st.number_input("Upfront Charges",      value=1500,   min_value=0,   step=100)
    term             = st.number_input("Loan Term (months)",   value=240,    min_value=1,   step=12)

st.subheader("Risk Factors")

col3, col4, col5 = st.columns(3)

with col3:
    credit_worthiness = st.selectbox("Credit Worthiness", ["Low", "Medium", "High"])
with col4:
    approv_in_adv     = st.selectbox("Approval in Advance", ["Pre-Approved", "Not Pre-Approved"])
with col5:
    dtir1             = st.slider("DTI Ratio (%)", min_value=0, max_value=100, value=30)

st.divider()

# ── Feature engineering (mirrors your Python pipeline exactly) ────────────────
def engineer_features(income, loan_amount, property_value, rate_of_interest,
                       upfront_charges, term, credit_worthiness, approv_in_adv, dtir1):

    loan_to_income_ratio   = loan_amount / (income + 1)
    LTV_percent            = (loan_amount / (property_value + 1)) * 100
    interest_burden        = (rate_of_interest * loan_amount) / (income + 1)
    upfront_charge_percent = (upfront_charges / (loan_amount + 1)) * 100
    long_term_flag         = 1 if term >= 240 else 0
    emi_to_income_ratio    = interest_burden / 10
    high_risk              = 1 if credit_worthiness == "Low" else 0

    approval_readiness_score = (
        (1 if approv_in_adv == "Pre-Approved" else 0) +
        (1 if credit_worthiness == "High" else 0) +
        (income / 100000)
    )

    dti_cat = (0 if dtir1 <= 20 else
               1 if dtir1 <= 35 else
               2 if dtir1 <= 50 else 3)

    df = pd.DataFrame([{
        "loan_to_income_ratio":   loan_to_income_ratio,
        "LTV_percent":            LTV_percent,
        "interest_burden":        interest_burden,
        "upfront_charge_percent": upfront_charge_percent,
        "long_term_flag":         long_term_flag,
        "high_risk":              high_risk,
        "approval_readiness_score": approval_readiness_score,
        "DTI_Category":           dti_cat,
        "emi_to_income_ratio":    emi_to_income_ratio,
    }])

    # Align feature columns to training model order to prevent XGBoost ValueError mismatch
    return df[[
        "loan_to_income_ratio",
        "LTV_percent",
        "interest_burden",
        "upfront_charge_percent",
        "long_term_flag",
        "emi_to_income_ratio",
        "high_risk",
        "approval_readiness_score",
        "DTI_Category",
    ]]

# ── Show computed features ─────────────────────────────────────────────────────
with st.expander("🔍 View Computed Features"):
    X = engineer_features(income, loan_amount, property_value, rate_of_interest,
                          upfront_charges, term, credit_worthiness, approv_in_adv, dtir1)
    st.dataframe(X.T.rename(columns={0: "Value"}).style.format("{:.4f}"), use_container_width=True)

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("🔮 Run Prediction", use_container_width=True, type="primary"):

    X = engineer_features(income, loan_amount, property_value, rate_of_interest,
                          upfront_charges, term, credit_worthiness, approv_in_adv, dtir1)

    prediction  = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    # ── Default / No Default ──────────────────────────────────────────────────
    if prediction == 1:
        st.error("⚠️ **Default Predicted** — High chance borrower will NOT repay loan")
    else:
        st.success("✅ **No Default Predicted** — Borrower likely to repay loan")

    # ── Probability bar ───────────────────────────────────────────────────────
    st.metric("Default Probability", f"{probability:.2%}")
    st.progress(float(probability))

    # ── Loan decision ─────────────────────────────────────────────────────────
    st.subheader("Loan Decision")

    if probability < 0.30:
        st.success("✅ **Loan Status: APPROVED**")
        st.caption("Low default risk. Borrower profile is strong.")
    elif probability < 0.60:
        st.warning("⚠️ **Loan Status: APPROVED WITH CONDITIONS**")
        st.caption("Moderate risk. Consider requiring collateral or a co-signer.")
    else:
        st.error("❌ **Loan Status: REJECTED**")
        st.caption("High default risk. Loan not recommended.")

    '''# ── Risk breakdown ────────────────────────────────────────────────────────
    st.subheader("Risk Factor Breakdown")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Loan-to-Income", f"{X['loan_to_income_ratio'].values[0]:.2f}",
                 delta="High" if X['loan_to_income_ratio'].values[0] > 3 else "OK",
                 delta_color="inverse")
    col_b.metric("LTV %", f"{X['LTV_percent'].values[0]:.1f}%",
                 delta="High" if X['LTV_percent'].values[0] > 80 else "OK",
                 delta_color="inverse")
    col_c.metric("Interest Burden", f"{X['interest_burden'].values[0]:.2f}",
                 delta="High" if X['interest_burden'].values[0] > 20 else "OK",
                 delta_color="inverse")

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Credit Risk",      "High" if high_risk else "Low")
    col_e.metric("DTI Category",     ["Low", "Moderate", "High", "Very High"][
                                      X['DTI_Category'].values[0]])
    col_f.metric("Approval Readiness", f"{X['approval_readiness_score'].values[0]:.2f}")'''
