from fastapi import FastAPI
import joblib
import numpy as np

# Initialize app
app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")

# Home route
@app.get("/")
def home():
    return {"message": "Customer Churn Model API Running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: list):
    try:
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)

        return {
            "prediction": int(prediction[0])
        }
    
    except Exception as e:
        return {"error": str(e)}