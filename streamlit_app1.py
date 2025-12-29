import streamlit as st
import re
import joblib
import pandas as pd

# ---------------- UI ----------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1732132966168-34cf0a39b840?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------- Preprocessing ----------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.joblib")
    vectorizer = joblib.load("tfidf.joblib")
    return model, vectorizer

model, vectorizer = load_model()

label_map = {
    0: "Negative 😞",
    1: "Neutral 😐",
    2: "Positive 😊"
}

# ---------------- App ----------------
st.header("🎮 Sentiment Analysis for Gaming Reviews")

with st.expander("✍️ Input Review"):
    user_input = st.text_area("Enter your review:")
    if user_input:
        clean_text = preprocess_text(user_input)
        vec = vectorizer.transform([clean_text])
        pred = model.predict(vec)[0]
        st.success(f"Predicted Sentiment: **{label_map[pred]}**")

with st.expander("📂 Analyse CSV"):
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'review_text' not in df.columns:
            st.error("CSV must contain 'review_text' column")
        else:
            df['clean_review'] = df['review_text'].astype(str).apply(preprocess_text)
            preds = model.predict(vectorizer.transform(df['clean_review']))
            df['prediction'] = [label_map[p] for p in preds]

            st.dataframe(df[['review_text', 'prediction']].head(10))
