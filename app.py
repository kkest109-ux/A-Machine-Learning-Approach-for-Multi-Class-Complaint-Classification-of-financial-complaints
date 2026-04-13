import streamlit as st
from joblib import load
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="📢",
    layout="wide"
)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = load(os.path.join(BASE_DIR, "model.pkl"))
    vectorizer = load(os.path.join(BASE_DIR, "vectorizer.pkl"))
    label_encoder = load(os.path.join(BASE_DIR, "label_encoder.pkl"))

    return model, vectorizer, label_encoder

model, vectorizer, le = load_models()

# ===============================
# LOAD DATA (FOR GRAPHS)
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("CFPB_Consumer_Complaints_2024.csv")
    df = df[['consumer_complaint_narrative', 'product']]
    df = df.dropna()
    return df

df = load_data()



# ===============================
# FUNCTIONS
# ===============================
def predict(text):
    text = text.lower()
    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]

    label = le.inverse_transform([pred])[0]
    confidence = np.max(probs)

    return label, confidence, probs

def get_top_k(probs, k=3):
    top_idx = probs.argsort()[-k:][::-1]
    return [(le.inverse_transform([i])[0], probs[i]) for i in top_idx]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### 📘 About")
st.sidebar.info(
    "ML Complaint Classifier\n\n"
    "Models: SVM, Logistic Regression, Naive Bayes\n"
    "Technique: Ensemble Learning\n"
    "Features: TF-IDF (n-grams)"
)

examples = [
    "My credit card payment is not processed",
    "Loan interest is too high",
    "Debt collection agency is harassing me",
    "Mortgage issue not resolved"
]

selected_example = st.sidebar.selectbox("🧪 Try Example", [""] + examples)

# ===============================
# MAIN UI
# ===============================
st.title("📢 Complaint Classification Dashboard")
st.markdown("### 🔍 Enter a complaint below")

user_input = st.text_area(
    "Complaint Text",
    value=selected_example,
    height=150
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Predict", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a complaint")
        else:
            label, confidence, probs = predict(user_input)

            # ===============================
            # RESULT
            # ===============================
            st.success(f"✅ Predicted Category: **{label}**")

            # Confidence
            st.markdown("### 📊 Confidence Score")
            st.progress(float(confidence))
            st.write(f"{confidence:.2f}")

            # ===============================
            # TOP-K TEXT OUTPUT
            # ===============================
            st.markdown("### 🔝 Top Predictions")
            top_preds = get_top_k(probs)

            for lbl, prob in top_preds:
                st.write(f"**{lbl}** → {prob:.2f}")

            # ===============================
            # GRAPH 1: PROBABILITY DISTRIBUTION
            # ===============================
            st.markdown("### 📈 Prediction Probabilities")

            fig2, ax2 = plt.subplots()

            # Get only labels used by model
            labels = le.inverse_transform(range(len(probs)))

            fig2, ax2 = plt.subplots()
            ax2.barh(labels, probs)

            ax2.set_xlabel("Probability")
            ax2.set_ylabel("Class")

            st.pyplot(fig2)


            # ===============================
            # GRAPH 2: TOP-K BAR GRAPH
            # ===============================
            st.markdown("### 📊 Top 3 Prediction Graph")

            top_labels = [str(x[0]) for x in top_preds]
            top_values = [float(x[1]) for x in top_preds]

            fig3, ax3 = plt.subplots()
            ax3.bar(top_labels, top_values)

            st.pyplot(fig3)

# ===============================
# RIGHT PANEL
# ===============================
with col2:
    st.markdown("### 📘 How it Works")
    st.write("""
    - Input text → cleaned & vectorized (TF-IDF)
    - ML models combined using Ensemble
    - Prediction based on probability
    """)

    st.markdown("### 📈 Model Info")
    st.write("""
    - Feature Extraction: TF-IDF (1-2 grams)
    - Models: SVM, Logistic Regression, Naive Bayes
    - Class Imbalance handled using SMOTE
    """)

    # ===============================
    # GRAPH 3: MODEL COMPARISON (STATIC)
    # ===============================
    st.markdown("### 📊 Model Accuracy Comparison")

    models = ["Logistic", "Naive Bayes", "SVM", "Ensemble"]
    scores = [0.82, 0.80, 0.85, 0.87]  # replace with your actual values

    fig4, ax4 = plt.subplots()
    ax4.bar(models, scores)

    st.pyplot(fig4)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("🚀 Built using Machine Learning + Streamlit")
