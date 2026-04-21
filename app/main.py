import sys
import os
import pandas as pd
import streamlit as st

# Add project root to Python path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from analysis import (
    load_data,
    get_kpis,
    fraud_by_specialty,
    fraud_by_insurance,
    fraud_by_claim_status,
    high_risk_providers,
    suspicious_claims,
)
from app.ai_assistant import AIFraudAssistant


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = df[col].round(2)
    return df


st.set_page_config(page_title="Healthcare Fraud Analyst", layout="wide")

st.title("Healthcare Fraud Detection Dashboard")
st.write("Small healthcare fraud analytics project built with Streamlit and Python.")

# Load data
df = load_data("data/healthcare_fraud_detection.csv")

# KPIs
kpis = get_kpis(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Claims", f"{kpis['total_claims']:,}")
col2.metric("Fraud Claims", f"{kpis['total_fraud_claims']:,}")
col3.metric("Fraud Rate", f"{kpis['fraud_pct']}%")
col4.metric("Total Claim Amount", f"${kpis['total_claim_amount']:,.2f}")
col5.metric("Total Approved Amount", f"${kpis['total_approved_amount']:,.2f}")

st.divider()

# Sidebar filters
st.sidebar.header("Filters")

specialty_options = ["All"] + sorted(df["Provider_Specialty"].dropna().unique().tolist())
insurance_options = ["All"] + sorted(df["Insurance_Type"].dropna().unique().tolist())
status_options = ["All"] + sorted(df["Claim_Status"].dropna().unique().tolist())

selected_specialty = st.sidebar.selectbox("Provider Specialty", specialty_options)
selected_insurance = st.sidebar.selectbox("Insurance Type", insurance_options)
selected_status = st.sidebar.selectbox("Claim Status", status_options)

filtered_df = df.copy()

if selected_specialty != "All":
    filtered_df = filtered_df[filtered_df["Provider_Specialty"] == selected_specialty]

if selected_insurance != "All":
    filtered_df = filtered_df[filtered_df["Insurance_Type"] == selected_insurance]

if selected_status != "All":
    filtered_df = filtered_df[filtered_df["Claim_Status"] == selected_status]

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered_df.head(50), use_container_width=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Fraud by Specialty",
        "Fraud by Insurance",
        "Fraud by Status",
        "High Risk & Suspicious Claims",
    ]
)

with tab1:
    st.subheader("Fraud by Provider Specialty")
    specialty_df = fraud_by_specialty(filtered_df)
    st.dataframe(format_dataframe(specialty_df), use_container_width=True)

with tab2:
    st.subheader("Fraud by Insurance Type")
    insurance_df = fraud_by_insurance(filtered_df)
    st.dataframe(format_dataframe(insurance_df), use_container_width=True)

with tab3:
    st.subheader("Fraud by Claim Status")
    status_df = fraud_by_claim_status(filtered_df)
    st.dataframe(format_dataframe(status_df), use_container_width=True)

with tab4:
    st.subheader("High Risk Providers")
    min_claims = st.slider("Minimum claims per provider", 5, 50, 20)
    provider_df = high_risk_providers(filtered_df, min_claims=min_claims)
    st.dataframe(format_dataframe(provider_df), use_container_width=True)

    st.subheader("Suspicious Claims")
    suspicious_df = suspicious_claims(filtered_df)
    st.dataframe(format_dataframe(suspicious_df.head(100)), use_container_width=True)

st.divider()
st.subheader("💬 AI Fraud Assistant")

if "ai_assistant" not in st.session_state:
    st.session_state.ai_assistant = AIFraudAssistant()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_tables" not in st.session_state:
    st.session_state.chat_tables = []

user_input = st.chat_input("Ask something about fraud data...", key="fraud_chat_input")

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    try:
        result = st.session_state.ai_assistant.ask(user_input)
        answer = result.get("answer", "No response generated.")
        table_data = result.get("table")

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

        if table_data:
            table_df = format_dataframe(pd.DataFrame(table_data))
        else:
            table_df = None

        st.session_state.chat_tables.append(table_df)

    except Exception as e:
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )
        st.session_state.chat_tables.append(None)

assistant_table_index = 0

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant":
            if assistant_table_index < len(st.session_state.chat_tables):
                table = st.session_state.chat_tables[assistant_table_index]
                if table is not None and not table.empty:
                    st.dataframe(table, use_container_width=True)
            assistant_table_index += 1