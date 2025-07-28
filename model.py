import joblib

# Load the trained model
def load_model():
    model = joblib.load("final_model.pkl")
    return model

# Predict using the model
def predict(model, features):
    return model.predict(features)
