# 🧠 Shaking Palsy (Parkinson's) Detection Using Voice + AI Chatbot

A Machine Learning-powered web app that detects signs of **Parkinson’s disease** from voice recordings and provides instant answers to user queries through an intelligent **AI medical chatbot**.

> 🎯 Built to help with early diagnosis and awareness for patients, researchers, and clinicians.

---

## 🚀 Live Demo

👉 **Try it here:** https://parkinson-disease-detection-with-chatbot.streamlit.app/


---

## 📌 Features

### 🎙️ Voice-Based Parkinson Detection
- Upload `.wav` audio file (sustained vowel sounds)
- Extracts 22 key vocal features like:
  - MFCCs
  - Jitter
  - Shimmer
  - Zero Crossing Rate
  - Harmonicity, etc.
- Predicts **Rigid Syndrome presence** using trained ML model
- Shows **Confidence Score**

### 🤖 AI-Powered Medical Chatbot
- Ask health-related questions like:
  - "What is shaking palsy?"
  - "Is Parkinson’s disease curable?"
  - "What are early symptoms?"
- Powered by **DistilBERT Question-Answering model (HuggingFace)**

---

## 🎯 Technologies Used

- **Frontend/UI**: Streamlit
- **Voice Processing**: Librosa, NumPy
- **ML Models**: XGBoost, SVM, RandomForest
- **Chatbot Model**: DistilBERT from HuggingFace Transformers
- **Language**: Python 3.11
- **Deployment**: Streamlit Cloud

---

## 📊 Machine Learning Pipeline

| Step                | Description                           |
|---------------------|---------------------------------------|
| Dataset             | UCI Parkinson Voice Dataset           |
| Features Extracted  | 22 audio-based features               |
| Models Tried        | SVM, RandomForest, XGBoost            |
| Final Model         | ✅ XGBoost (~93% Accuracy)             |
| Format              | Model saved as `final_model.pkl`      |

---

## 🗂️ Project Structure

Parkinson_Chatbot_ML/
├── app.py # Main Streamlit app
├── model.py # ML model load/predict logic
├── chatbot.py # AI chatbot (DistilBERT)
├── final_model.pkl # Trained XGBoost model
├── requirements.txt # All dependencies
├── .gitignore # Ignores venv/, pycache, etc.
├── README.md # This file
└── src/
└── feature_extraction.py # Audio processing functions


---

## 🧪 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Farhan2116/Parkinson_Chatbot_ML.git
cd Parkinson_Chatbot_ML

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
💡 Sample Test Audio
Use the provided parkinsons_sample.wav or upload your own .wav voice file for testing.

👨‍💻 Author
Shaik Farhan
AI & Data Science Enthusiast | 2025 Graduate in CSE (AI/ML)
📫 LinkedIn

