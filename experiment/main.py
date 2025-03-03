from fastapi import FastAPI
import pandas as pd 
import numpy as np
from thefuzz import fuzz,process
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

df = pd.read_csv('5.urldata.csv', usecols=["Domain"],low_memory=False)
df["Domain"] = df["Domain"].str.lower()


def find_best_match(phishing_domain, legit_domains):
    # Vectorized fuzzy matching to get the best match
    best_match = process.extractOne(phishing_domain, legit_domains, scorer=fuzz.ratio)
    return best_match  # Returns (best_match_domain, score)

# Convert column to a NumPy array for faster access
legit_domains = df["Domain"].values



app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:5174"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.get("/recommend")
def recommend(phishing_domain: str):
    best_match = find_best_match(phishing_domain, legit_domains)
    return {"input": phishing_domain, "recommended": best_match[0]}
