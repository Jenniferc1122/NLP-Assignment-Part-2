import streamlit as st
import re
import joblib
import pandas as pd
import emoji

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
    # convert to lowercase
    text = text.lower()
    # convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))
    # replace underscores in emoji text with spaces
    text = text.replace("_", " ")
    # remove punctuation & special characters
    text = re.sub(r'[^a-z_\s]', '', text)
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
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

tab1, tab2 = st.tabs([
    "✍️ Sentiment Analyser",
    "📖 System Workflow"])

with tab1:
    st.subheader("📊 Model Performance")
    
    st.metric(
        label="F1 Score",
        value=f"{f1_score_value * 100:.2f}%"
    )

    # ---------------- Function to do single sentiment prediction ----------------
    
    with st.expander("✍️ Input Review"):
        user_input = st.text_area("Enter your review:")
        if user_input:
            clean_text = preprocess_text(user_input)
            pred = model.predict([clean_text])[0]
            st.success(f"Predicted Sentiment: **{label_map[pred]}**")

    # ---------------- Function to upload review in bulk with CSV ----------------
    
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
    
        st.markdown("---") 
    
        # ---------------- Function to upload sample CSV ----------------
        
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

with tab2:
    st.subheader("🧩 System Workflow")
    st.markdown("""    This section explains how the sentiment analyser system works internally.""")

    # ---------------- Workflow Steps  ----------------
    
    with st.expander("Step 1 – Text Preprocessing"):
        st.markdown("""
        - Text is converted to lowercase to ensure consistency.
        - Emoji is converted to text form.
        - Puctuations, special characters and extra white spaces are removed.
        """)

    with st.expander("Step 2 – Data Labelling"):
        st.markdown("""
        - Label encoding to convert sentiment category into numerical.
            - Positive reviews are labelled as 2.
            - Neutral reviews are labelled as 1.
            - Negative reviews are labelled as 0.
        """)

    with st.expander("Step 3 – Data Splitting"):
        st.markdown("""
        - 70% of the data is used for training.
        - 30% of the data is used for testing.
        """)

    with st.expander("Step 4 – Model Training and Hyperparameter Tuning"):
        st.markdown("""
        - Pipeline is created to combine TF-IDF with Linear SVC into a single workflow.
        - Hyperparameter tuning is performed using RandomizedSearchCV to find the best model parameters.
        """)

    with st.expander("Step 5 – Model Evaluation"):
        st.markdown("""
        - The final model is evaluated using F1 score metric.
        - Classification report is generated to assess model performance across different sentiment categories.
        """)

    with st.expander("Step 6 – Model Saving"):
        st.markdown("""
        - The best performing model is saved using Joblib for streamlit integration.
        """)

    with st.expander("⚠️ System Limitations"):
        st.markdown("""
        Current Limitations
        - Unable to predict sentiment of all emojis due to the limited emoji content in the training data.
        - Sarcasm detection.
        - Field specific, weaker sentiment prediction when it is used to predict sentiment in other field.
        - Only applicable to reviews in English.
        """)

    with st.expander("🚀 Future Improvements"):
        st.markdown(""" 
        - Explore Deep Learning models to improve sentiment prediction performance.
        - Enhance the preprocessing step with language translation function (Cross Language Sentiment Analysis)
        """)





    
        


