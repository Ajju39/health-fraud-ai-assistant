import streamlit as st
import pandas as pd
from pathlib import Path
from ai_assistant import ask_health_fraud_assistant

st.set_page_config(page_title="Health Fraud AI Assistant", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

st.title("🤖 Health Fraud AI Assistant")
st.caption("Fraud analytics dashboard + AI assistant")

# ---------------- Dashboard Section ----------------
st.subheader("📊 Fraud Dashboard")

claims_path = DATA_DIR / "claims.csv"

if claims_path.exists():
    claims = pd.read_csv(claims_path)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", len(claims))

    if "fraud_flag" in claims.columns:
        col2.metric("Fraud Claims", int(claims["fraud_flag"].sum()))
    else:
        col2.metric("Fraud Claims", "N/A")

    if "provider_id" in claims.columns:
        col3.metric("Providers", claims["provider_id"].nunique())
    else:
        col3.metric("Providers", "N/A")

    st.subheader("📄 Claims Data")
    st.dataframe(claims, use_container_width=True)

    if "provider_id" in claims.columns:
        st.subheader("🏥 Claims by Provider")
        provider_counts = claims["provider_id"].value_counts().reset_index()
        provider_counts.columns = ["provider_id", "claim_count"]
        st.bar_chart(provider_counts.set_index("provider_id"))

else:
    st.warning("claims.csv not found. Please check your data folder path.")

st.divider()

# ---------------- AI Bot Section ----------------
st.subheader("🤖 Ask AI Fraud Assistant")

st.markdown("""
### Example Queries
- Check claim C0000001
- Is claim C0000002 safe?
- Is claim C0000002 risky?
- Analyze provider P0140
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        answer = ask_health_fraud_assistant(user_input)
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})