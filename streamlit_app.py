import pickle
import streamlit as st
import cleantext
import re
import nltk
import joblib

#--- Streamlit UI ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1732132966168-34cf0a39b840?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

import nltk
from nltk.corpus import stopwords

# Download only if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove punctuation & special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # tokenisation
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords & stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

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

#--- Streamlit App ---

st.header("**Sentiment Analysis for Gaming Review**")
with st.expander("Input Review"):
    user_input = st.text_area("Enter your review here:")
    if user_input:
        clean_text = preprocess_text(user_input)
        vec_text = vectorizer.transform([clean_text])
        prediction = model.predict(vec_text)[0]
        sentiment = label_map[prediction]
        st.success(f"Predicted Sentiment: {sentiment}")

    #pre = st.text_input("Clean Review:")
    #if pre:
    #    st.write(cleantext.clean(pre, clean_all=False, extra_spaces=True, punct=True, lowercase=True, stopwords=True, numbers=True))

with st.expander("Analyse CSV"):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.write(df.head())

        if 'review_text' in df.columns:
            df['clean_review'] = df['review_text'].apply(preprocess_text)
            # Predict
            preds = model.predict(
                vectorizer.transform(df['clean_review'])
            )

            # Convert numeric labels to text
            df['prediction'] = [label_map[p] for p in preds]

            st.write("Prediction results:")
            st.write(df[['review_text', 'Prediction']].head())

        else:
            st.error("CSV must contain a 'review_text' column")
