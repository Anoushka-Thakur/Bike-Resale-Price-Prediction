# Used Bike Resale Price Prediction App
### Project Overview

This project is a machine learning–powered web application that predicts the resale price of used bikes based on key features such as brand, model, kilometers driven, mileage, engine capacity, and ownership details.

The goal is to help users make data-driven pricing decisions when buying or selling second-hand bikes.

### Problem Statement

The resale value of bikes varies widely depending on usage, specifications, and brand perception.
Manually estimating a fair price is difficult and often inaccurate.

This application solves that problem by:

Analyzing historical bike data

Applying feature engineering and machine learning

Providing instant price predictions through an interactive UI

### Solution Approach

1. Data Cleaning & Preprocessing

2. Removed inconsistencies and handled missing values

3. Cleaned numeric columns like price, mileage, engine capacity, and kilometers driven

### Feature Engineering

1. Converted categorical features using encoding techniques

2. Selected important features impacting resale price

### Model Building

1. Trained regression models using Scikit-learn

2. Applied log transformation on target variable for better prediction stability

### Model Deployment

1. Built an interactive Streamlit web application

2. Deployed on Streamlit Cloud

### Tech Stack

Python

Pandas & NumPy – Data cleaning & analysis

Scikit-learn – Model training

Matplotlib & Seaborn – Exploratory Data Analysis

Streamlit – Web app & deployment

### Features of the App

1. Dropdown-based selection for:

Bike brand & model

Ownership type

2. Slider inputs for:

Kilometers driven

Mileage

Engine capacity

Real-time resale price prediction

Clean and user-friendly UI

📁 Project Structure
Used-Bike-Resale-Price-Prediction/
1. app.py                  # Streamlit app
2.  bikes.csv               # Dataset
3. requirements.txt        # Dependencies
4. runtime.txt             # Python version for deployment
5. Used Bike Prices - Feature Engineering and EDA.ipynb
6. README.md

### How to Run Locally
pip install -r requirements.txt
streamlit run app.py

### Live Demo

👉 https://bike-resale-price-prediction-kyu4rdee5cwkynqd8c5ucv.streamlit.app/


### Author

Anoushka Thakur
Data Analyst | Machine Learning Enthusiast
Intern at Unified Mentor
