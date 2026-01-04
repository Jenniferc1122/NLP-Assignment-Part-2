import streamlit as st
import re
import joblib
import pandas as pd

# ---------------- UI ----------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

page_bg_img = """
<style>
/* Background image on the app container with a dark overlay for contrast */
[data-testid="stAppViewContainer"] {
    position: relative;
    background-image: url("https://free-vectors.net/_ph/6/805542910.jpg");
    background-size: cover;
}

}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0)
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------------- Show f1 Score ----------------

@st.cache_resource
def load_metrics():
    f1 = joblib.load("model_f1_score.joblib")
    return f1

f1_score_value = load_metrics()

# ---------------- Preprocessing ----------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = joblib.load("final_sentiment_model.joblib")
    return model

model = load_model()

label_map = {
    0: "Negative 😞",
    1: "Neutral 😐",
    2: "Positive 😊"
}

# ---------------- App ----------------
st.header("🎮 Sentiment Analysis for Gaming Reviews")

sample_csv_url = "https://raw.githubusercontent.com/Jenniferc1122/NLP-Assignment-Part-2/refs/heads/master/forsample.csv"

st.subheader("📊 Model Performance")

st.metric(
    label="F1 Score",
    value=f"{f1_score_value * 100:.2f}%"
)

with st.expander("✍️ Input Review"):
    user_input = st.text_area("Enter your review:")
    if user_input:
        clean_text = preprocess_text(user_input)
        pred = model.predict([clean_text])[0]
        st.success(f"Predicted Sentiment: **{label_map[pred]}**")

with st.expander("📂 Analyse CSV"):
    st.markdown("### Try a sample or upload your own CSV")
    # Uploader (top)
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if 'review_text' not in df.columns:
            st.error("CSV must contain 'review_text' column")
        else:
            df['clean_review'] = df['review_text'].astype(str).apply(preprocess_text)
            preds = model.predict(df['clean_review'])
            df['prediction'] = [label_map[p] for p in preds]

            st.dataframe(df[['review_text', 'prediction']].head(10))

    st.markdown("---")  # separator between uploader and sample action

    # Sample CSV (below)
    if st.button("Try with Sample CSV"):
        df = pd.read_csv(sample_csv_url)

        if 'review_text' not in df.columns:
            st.error("Sample CSV must contain 'review_text' column")
        else:
            df['clean_review'] = df['review_text'].astype(str).apply(preprocess_text)
            preds = model.predict(df['clean_review'])
            df['prediction'] = [label_map[p] for p in preds]
            st.success("Sample file analysed successfully!")
            st.dataframe(df[['review_text', 'prediction']].head(10))

