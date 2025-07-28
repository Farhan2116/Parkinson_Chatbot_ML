import streamlit as st
import numpy as np
import soundfile as sf
import os

from model import load_model, predict
from src.feature_extraction import extract_features
from chatbot import get_bot_response

# Set Streamlit app title
st.set_page_config(page_title="Shaking Palsy Detection", layout="centered")
st.title("🧠 Shaking Palsy Detection using Voice")

# File uploader
uploaded_file = st.file_uploader("🎙️ Upload a Voice File (.wav or .mp3)", type=["wav", "mp3"])

# Load the trained model
model = load_model()

# Prediction logic
if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        # Extract features
        features = extract_features("temp.wav").reshape(1, -1)

        # Predict
        prediction = predict(model, features)
        confidence = round(np.max(model.predict_proba(features)) * 100, 2)

        # Show result
        if prediction[0] == 1:
            st.error("🔴 Rigid Syndrome Detected")
        else:
            st.success("🟢 Voice is Healthy")
        st.info(f"Confidence: {confidence}%")

        # Remove temp file
        os.remove("temp.wav")

    except Exception as e:
        st.error(f"Error processing the file: {e}")

# ------------------ Chatbot Section ------------------
st.markdown("---")
st.header("💬 Ask Our Health Bot")
user_question = st.text_input("Ask a question about Parkinson's or shaking palsy")

if user_question:
    with st.spinner("Thinking..."):
        answer = get_bot_response(user_question)
        st.success(f"🤖 Answer: {answer}")
