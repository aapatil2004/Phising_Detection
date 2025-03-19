from urllib.parse import urlparse
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from thefuzz import fuzz, process
import whois
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware

from URLFeatureExtraction import featureExtraction

# Load legitimate domains for fuzzy matching
df = pd.read_csv('5.urldata.csv', usecols=["Domain"], low_memory=False)
df["Domain"] = df["Domain"].str.lower()
legit_domains = df["Domain"].values

# Load trained phishing detection model
with open("XGBoostClassifier.pickle.dat", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Phishing detection API is running!"}


# Request model for /predict endpoint
class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict_phishing(request: URLRequest):
    features = featureExtraction(request.url)
    print(f"Extracted Features: {features}")  # Debugging
    prediction = model.predict(np.array([features]))[0]
    result = "Phishing" if prediction == 1 else "Legitimate"
    return {"url": request.url, "prediction": result}
    # """Predict if a website is phishing or legitimate using features from CSV."""
    
    # # Extract features for the given domain
    # domain_data = df[df["Domain"] == request.url]
    

    # # Drop non-feature columns (assuming first is Domain, last is Label)
    # features = domain_data.iloc[:, 1:-1].values  # Keep only feature columns

    # # Make prediction
    # prediction = model.predict(features)[0]
    # result = "Phishing" if prediction == 1 else "Legitimate"

    # return {"url": request.url, "prediction": result}
    
    

@app.get("/recommend")
def recommend_legit_website(phishing_domain: str):
    """Find the closest legitimate website for a phishing domain."""
    best_match = process.extractOne(phishing_domain, legit_domains, scorer=fuzz.ratio)
    return {"phishing_domain": phishing_domain, "recommended_legit": best_match[0] if best_match else None}