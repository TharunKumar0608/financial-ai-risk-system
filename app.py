import os
import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Financial AI", layout="wide")

st.title("💡 AI Financial Health Intelligence System")
st.markdown("Real-time Financial Diagnostics & Predictive Risk Analysis")

# ---------------- LOAD MODEL ----------------
try:
    import os

    st.write("Current Directory:", os.getcwd())
    st.write("Files in directory:", os.listdir())

    model = pickle.load(open("financial_risk_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📊 Financial Inputs")

age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income", 1000, 1000000, 50000)
expenses = st.sidebar.number_input("Monthly Expenses", 0, 500000, 20000)
savings = st.sidebar.number_input("Savings Balance", 0, 1000000, 30000)
investment = st.sidebar.number_input("Investment Balance", 0, 1000000, 20000)
loan = st.sidebar.number_input("Loan Balance", 0, 1000000, 10000)
credit_card = st.sidebar.number_input("Credit Card Balance", 0, 500000, 5000)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)

st.sidebar.header("📈 Behavioral Inputs")

missed_payments = st.sidebar.slider("Missed Payments (6 months)", 0, 10, 1)
emi_delays = st.sidebar.slider("EMI Delays", 0, 10, 1)
credit_utilization = st.sidebar.slider("Credit Utilization (%)", 0, 100, 40)

# ---------------- FEATURE ENGINEERING ----------------
if income == 0:
    st.warning("Income cannot be zero")
    st.stop()

expense_ratio = min(expenses / income, 1)
savings_rate = min(savings / income, 1)
investment_rate = min(investment / income, 1)
credit_util = credit_utilization / 100

# ---------------- PREDICT BUTTON ----------------
if st.sidebar.button("🚀 Predict Financial Risk"):

    input_data = np.array([[
        age,
        income,
        expenses,
        savings,
        credit_card,
        loan,
        investment,
        credit_score,
        missed_payments,
        emi_delays,
        expense_ratio,
        savings_rate,
        investment_rate,
        credit_util
    ]])

    # SCALE DATA
    scaled_data = scaler.transform(input_data)

    # MODEL PREDICTION
    prediction = model.predict(scaled_data)[0]

    # ---------------- RISK CATEGORY ----------------
    if prediction == 0:
        risk = "🟢 Financially Stable"
    elif prediction == 1:
        risk = "🟡 Moderate Risk Exposure"
    else:
        risk = "🔴 High Financial Vulnerability"

    # ---------------- HEALTH SCORE ----------------
    health_score = int(
    (savings_rate * 40) +
    (investment_rate * 30) +
    ((1 - expense_ratio) * 30)
)

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 Prediction Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Financial Health Score", f"{health_score}/100")
    col2.metric("⚠ Risk Category", risk)

    if "Stable" in risk:
        col3.success("🟢 Strong Financial Condition")
    elif "Moderate" in risk:
        col3.warning("🟡 Needs Improvement")
    else:
        col3.error("🔴 High Risk Detected")

    st.divider()

    # ---------------- GAUGE ----------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Financial Health Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "#ff4d4d"},
                {'range': [40, 70], 'color': "#ffd633"},
                {'range': [70, 100], 'color': "#33cc33"},
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # ---------------- CHART ----------------
    chart_data = {
        "Category": ["Income", "Expenses", "Savings", "Investment"],
        "Amount": [income, expenses, savings, investment]
    }

    fig = px.bar(
        x=chart_data["Category"],
        y=chart_data["Amount"],
        color=chart_data["Category"],
        text=chart_data["Amount"],
        title="📈 Financial Distribution"
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RATIOS ----------------
    st.subheader("📊 Financial Ratios")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Expense Ratio", round(expense_ratio, 2))
    col2.metric("Savings Rate", round(savings_rate, 2))
    col3.metric("Investment Rate", round(investment_rate, 2))
    col4.metric("Credit Utilization", round(credit_util, 2))

    # ---------------- AI RECOMMENDATIONS ----------------
    st.subheader("🤖 AI Financial Recommendations")

    if "Vulnerability" in risk:

        st.error("🔴 Poor Financial Condition")

        st.write("⚠ Immediate Actions:")
        st.write("• Reduce expenses immediately")
        st.write("• Avoid new loans")
        st.write("• Clear existing debt")

        st.write("💡 Improvement:")
        st.write("• Build emergency fund")
        st.write("• Increase savings to 20%")
        st.write("• Improve payment discipline")

    elif "Moderate" in risk:

        st.warning("🟡 Moderate Financial Risk")

        st.write("⚠ Improve:")
        st.write("• Optimize expenses")
        st.write("• Reduce credit usage")
        st.write("• Track loans")

        st.write("💡 Grow:")
        st.write("• Increase savings")
        st.write("• Start investments")
        st.write("• Diversify income")

    else:

        st.success("🟢 Excellent Financial Health")

        st.write("✅ Keep it up!")
        st.write("• Continue saving & investing")
        st.write("• Maintain discipline")

        st.write("🚀 Growth:")
        st.write("• Explore advanced investments")
        st.write("• Plan long-term wealth")

    # ---------------- PERSONALIZED INSIGHTS ----------------
    st.subheader("📌 Personalized Insights")

    if expense_ratio > 0.6:
        st.warning("High expense ratio")

    if savings_rate < 0.2:
        st.info("Savings below recommended level")

    if credit_util > 0.5:
        st.warning("High credit utilization")

    if missed_payments > 2:
        st.error("Too many missed payments")

    if emi_delays > 2:
        st.error("Frequent EMI delays")

    # ---------------- EXPLAINABLE AI ----------------
    st.subheader("🔍 Explainable AI Insights")

    st.write(f"Expense Ratio: {round(expense_ratio,2)}")
    st.write(f"Savings Rate: {round(savings_rate,2)}")
    st.write(f"Investment Rate: {round(investment_rate,2)}")
    st.write(f"Credit Utilization: {round(credit_util,2)}")
