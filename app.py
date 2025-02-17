import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import subprocess
from streamlit_ace import st_ace
import os
from dotenv import load_dotenv
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN

# Load environment variables from app.env
load_dotenv("app.env")

# Get API key from environment variables
key = os.getenv("GEMINI_API_KEY")

# Check if API key exists
if not key:
    st.error("Error: Gemini API key not found. Set GEMINI_API_KEY in app.env.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

st.title("🔍 AI-Powered Data Analysis & ML Toolkit")

# Sidebar for different tools
option = st.sidebar.radio(
    "Choose a tool:",
    ["Python Code Review", "MySQL Query Review", "Data Analysis", "Machine Learning"]
)

# Python Code Review
if option == "Python Code Review":
    st.subheader("Python Code Review & Execution")

    code = st_ace(language="python", theme="monokai", height=300)
    
    if st.button("Run Code"):
        with open("temp.py", "w") as f:
            f.write(code)
        
        result = subprocess.run(["python3", "temp.py"], capture_output=True, text=True)
        st.code(result.stdout if result.stdout else result.stderr)

    if st.button("Review Code"):
        response = model.generate_content(f"Review this Python code and suggest improvements:\n{code}")
        st.markdown(response.text)
