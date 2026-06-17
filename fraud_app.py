import requests
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.express as px
import os
from dotenv import load_dotenv

#load_dotenv()  # reads .env
#api_key = os.getenv("GEMINI_API_KEY")
#import google.generativeai as genai
#genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

#from google import genai
#client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

#model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
#st.write(st.secrets["GEMINI_API_KEY"])

#st.write(api_key)
#from openai import OpenAI
#client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("bank_transactions_data_2.csv")

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

#converting date features into datetime format
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], dayfirst=True)
df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], dayfirst=True)

df = df.sort_values(["AccountID", "TransactionDate"])

#creating new columns txnGapMinutes
df["TxnGapMinutes"] = (
    df.groupby("AccountID")["TransactionDate"]
      .diff()
      .dt.total_seconds() / 60
)

#filling null values in txnGapMinutes
df["IsFirstTxn"] = df["TxnGapMinutes"].isna().astype(int) #for unique account IDs
df["TxnGapMinutes"] = df["TxnGapMinutes"].fillna(df["TxnGapMinutes"].median())
df = df.reset_index(drop=True)

#creating new columns
df["TxnHour"] = df["TransactionDate"].dt.hour
df["TxnWeekday"] = df["TransactionDate"].dt.weekday
df["IsWeekend"] = df["TxnWeekday"].isin([5, 6]).astype(int)

#creating new columns Amount-to-balance ratio
df["Amount_to_Balance_Ratio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1)
df["Debit_Amount_Ratio"] = 0
df["Credit_Amount_Ratio"] = 0

#creating separate columns Amount-to-balance ratio for DEBIT and CREDIT types
df.loc[df["TransactionType"] == "Debit", "Debit_Amount_Ratio"] = (
    df["TransactionAmount"] / (df["AccountBalance"] + 1)
)

df.loc[df["TransactionType"] == "Credit", "Credit_Amount_Ratio"] = (
    df["TransactionAmount"] / (df["AccountBalance"] + 1)
)

#creating new columns HighLoginAttempts
df["HighLoginAttempts"] = (df["LoginAttempts"] > 2).astype(int)

#creating new columns LongDurationFlag
df["LongDurationFlag"] = (
    df["TransactionDuration"] > df["TransactionDuration"].quantile(0.95)
).astype(int)

# -----------------------------
# CREATE MODEL DATASET (df_model)
# -----------------------------
features = [
    "TransactionAmount",
    "Amount_to_Balance_Ratio",
    "TxnGapMinutes",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance",
    "HighLoginAttempts",
    "LongDurationFlag",
    "IsFirstTxn"
]
df_model = df[features].copy()

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
iso_model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# SCALING
# -----------------------------
scaled_data = scaler.transform(df_model)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = iso_model.predict(scaled_data)
df["AnomalyLabel"] = prediction
df["IsAnomaly"] = (df["AnomalyLabel"] == -1).astype(int)

#st.write(df.columns)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Filters")

# Date filter
start_date = st.sidebar.date_input(
    "Start Date", df["TransactionDate"].min()
)
end_date = st.sidebar.date_input(
    "End Date", df["TransactionDate"].max()
)

# Amount slider
min_amount, max_amount = float(df["TransactionAmount"].min()), float(df["TransactionAmount"].max())

amount_range = st.sidebar.slider(
    "Transaction Amount Range",
    min_value=min_amount,
    max_value=max_amount,
    value=(min_amount, max_amount)
)

# Apply filters
filtered_df = df[
    (df["TransactionDate"] >= pd.to_datetime(start_date)) &
    (df["TransactionDate"] <= pd.to_datetime(end_date)) &
    (df["TransactionAmount"] >= amount_range[0]) &
    (df["TransactionAmount"] <= amount_range[1])
]

# --------------------------------------------------
# METRICS
# --------------------------------------------------
total_transactions = len(filtered_df)
total_fraud = filtered_df["IsAnomaly"].sum()
fraud_percentage = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_transactions)
col2.metric("Fraud Transactions", total_fraud)
col3.metric("Fraud %", f"{fraud_percentage:.2f}%")

st.markdown("---")

col1, col2 = st.columns([1, 2])
# -----------------------------
# PIE CHART
# -----------------------------
with col1:
    fig1 = px.pie(
        df,
        names="IsAnomaly",
        title="Fraud vs Normal Distribution",
        labels={"IsAnomaly": "Transaction Type"}
    )
    st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# TRANSACTION AMOUNT DISTRIBUTION
# -----------------------------
with col2:
    fig2 = px.histogram(
        df,
        x="TransactionAmount",
        color="IsAnomaly",
        nbins=50,
        title="Transaction Amount Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# SHOW FRAUD TABLE
# -----------------------------

# -----------------------------------
# FILTER FRAUD DATA
# -----------------------------------
fraud_df = df[df["IsAnomaly"] == 1]

if not fraud_df.empty:

    fraud_df = fraud_df.sort_values(by="TransactionID")

    st.subheader("⚠ Fraud Investigation Dashboard")

    # ===================================
    # 🔍 TOP FILTER SECTION (FULL WIDTH)
    # ===================================
    st.markdown("### 🔍 Filters")

    min_amt = float(fraud_df["TransactionAmount"].min())
    max_amt = float(fraud_df["TransactionAmount"].max())

    amount_range = st.slider(
        "Transaction Amount Range",
        min_value=min_amt,
        max_value=max_amt,
        value=(min_amt, max_amt)
    )

    hour_filter = st.multiselect(
        "Transaction Hour",
        options=sorted(fraud_df["TxnHour"].unique()),
        default=sorted(fraud_df["TxnHour"].unique())
    )

    weekend_filter = st.multiselect(
        "Day Type",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "Weekend" if x == 1 else "Weekday"
    )

    # ===================================
    # APPLY FILTERS
    # ===================================
    filtered_df = fraud_df[
        (fraud_df["TransactionAmount"] >= amount_range[0]) &
        (fraud_df["TransactionAmount"] <= amount_range[1]) &
        (fraud_df["TxnHour"].isin(hour_filter)) &
        (fraud_df["IsWeekend"].isin(weekend_filter))
    ]

    filtered_df = filtered_df.reset_index(drop=True)

    # ===================================
    # BELOW FILTERS → 2 COLUMN LAYOUT
    # ===================================
    col1, col2 = st.columns([2, 1])

    if not filtered_df.empty:

        # -------------------------
        # COLUMN 1 → TABLE
        # -------------------------
        with col1:
            st.markdown("### 📋 Fraud Transactions")

            selected_txn_id = st.selectbox(
                "🔎 Select Transaction ID",
                filtered_df["TransactionID"]
            )

            st.dataframe(filtered_df, hide_index=True)

        # -------------------------
        # COLUMN 2 → EXPLANATION
        # -------------------------
        with col2:
            st.markdown("### 🤖 Fraud Explanation")

            selected_row = filtered_df[
                filtered_df["TransactionID"] == selected_txn_id
            ].iloc[0]

            risk_points = []
            risk_score = 0

            # Rule 1 – High amount
            if selected_row["TransactionAmount"] > df["TransactionAmount"].mean():
                risk_points.append("Unusually high transaction amount compared to average.")
                risk_score += 1

            # Rule 2 – Unusual hour
            if selected_row["TxnHour"] < 6 or selected_row["TxnHour"] > 22:
                risk_points.append("Transaction occurred during unusual hours.")
                risk_score += 1

            # Rule 3 – Weekend anomaly
            if selected_row["IsWeekend"] == 1:
                risk_points.append("Weekend transaction pattern deviation.")
                risk_score += 1

            # Rule 4 – Age anomaly
            if selected_row["CustomerAge"] < 21 or selected_row["CustomerAge"] > 70:
                risk_points.append("Transaction behavior unusual for this age segment.")
                risk_score += 1

            # Display reasons
            if risk_points:
                st.markdown("#### 🚩 Key Risk Indicators")
                for r in risk_points:
                    st.write("•", r)
            else:
                st.write("Flagged due to overall anomaly pattern detected by ML model.")

            # Risk Level
            st.markdown("#### 📊 Risk Level")

            if risk_score >= 4:
                st.error("High Risk 🚨")
            elif risk_score >= 2:
                st.warning("Medium Risk ⚠")
            else:
                st.info("Low Risk")

    else:
        st.warning("No transactions match selected filters.")

else:
    st.write("No fraud transactions detected.")


#------------------------------
# FRAUD TREND
# -----------------------------

st.subheader("📈 Fraud Trend Over Time")

fraud_trend = (
    df[df["IsAnomaly"] == 1]
    .groupby(df["TransactionDate"].dt.date)
    .size()
)

fig, ax = plt.subplots(figsize=(6,3))
fraud_trend.plot(ax=ax)

ax.set_title("Fraud Cases Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Fraud Count")

st.pyplot(fig)

#-----------------------
