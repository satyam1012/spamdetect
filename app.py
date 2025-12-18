import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e3f2fd, #ede7f6);
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background-color: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
}
h1 {
    text-align: center;
    color: #4a148c;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = joblib.load("model.joblib")
vectorizer = joblib.load("scaled.joblib")

# ---------------- UI ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check whether it is **Spam** or **Ham (Not Spam)**.")

message = st.text_area(
    "✍️ Type your message",
    height=150,
    placeholder="Win a free iPhone by clicking this link..."
)

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        X_input = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(X_input)[0]

        # Probability
        probabilities = model.predict_proba(X_input)[0]
        spam_prob = probabilities[1] * 100
        ham_prob = probabilities[0] * 100

        # Result
        if prediction == "spam":
            st.error(f"🚨 **SPAM DETECTED**\n\nConfidence: **{spam_prob:.2f}%**")
        else:
            st.success(f"✅ **HAM (Not Spam)**\n\nConfidence: **{ham_prob:.2f}%**")

        # ---------------- Probability Graph ----------------
        st.subheader("📊 Prediction Probability")

        prob_df = pd.DataFrame({
            "Category": ["Ham", "Spam"],
            "Probability (%)": [ham_prob, spam_prob]
        })

        st.bar_chart(prob_df.set_index("Category"))

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("🔐 ML-based Spam Detection using NLP & Streamlit")
