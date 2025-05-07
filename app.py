import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LEN = 300

# Load model and tokenizer
try:
    model = tf.keras.models.load_model("fake_news_model.keras", compile=False)
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
except Exception as e:
    st.error("Error loading model or tokenizer:")
    st.exception(e)
    st.stop()

# App title
st.title("📰 Fake News Detector")

# Input text
user_input = st.text_area("Enter news article text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # Predict
        prediction = model.predict(padded)[0][0]
        label = "🟢 Real News" if prediction > 0.5 else "🔴 Fake News"

        # Show result
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence: {prediction:.2f}**")
