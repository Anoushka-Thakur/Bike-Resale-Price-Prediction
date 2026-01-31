import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Used Bike Price Predictor",
    page_icon="🏍️",
    layout="wide"
)
# --------------------------------------------------
# TITLE (INLINE STYLING FIX)
# --------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#ff7a18;">
        🏍️ Used Bike Price Prediction
    </h1>
    <p style="text-align:center; color:#ff7a18; font-size:18px;">
        Predict resale value using machine learning
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL & ENCODERS
# --------------------------------------------------
#  Load data
@st.cache_data
def load_data():
    return pd.read_csv("bikes.csv")

data = load_data()

#  Clean numeric columns (VERY IMPORTANT)
numeric_cols = ['kms_driven', 'mileage', 'power']

for col in numeric_cols:
    data[col] = (
        data[col]
        .astype(str)
        .str.lower()
        .str.replace(r'[^0-9\.]', '', regex=True)   # keep digits & dots
        .str.replace(r'\.(?=.*\.)', '', regex=True)  # remove extra dots
        .replace('', np.nan)
        .astype(float)
    )

data.dropna(subset=numeric_cols, inplace=True)


data.dropna(subset=numeric_cols, inplace=True)

#  Encode categorical columns
cat_cols = ['model_name', 'location', 'owner']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le


# --------------------------------------------------
# SIDEBAR — USER INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 Bike Details")

model_name_input = st.sidebar.selectbox(
    "Bike Model",
    encoders["model_name"].classes_,
    key="model_select"
)

location_input = st.sidebar.selectbox(
    "Location",
    encoders["location"].classes_,
    key="location_select"
)

owner_input = st.sidebar.selectbox(
    "Owner Type",
    encoders["owner"].classes_,
    key="owner_select"
)

bike_age = st.sidebar.slider(
    "Bike Age (Years)",
    0, 20, 5,
    key="bike_age_slider"
)

kms_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=200000,
    value=20000,
    key="kms_input"
)

mileage = st.sidebar.number_input(
    "Mileage (kmpl)",
    min_value=5.0,
    max_value=100.0,
    value=35.0,
    key="mileage_input"
)

power = st.sidebar.number_input(
    "Power (bhp)",
    min_value=5.0,
    max_value=100.0,
    value=20.0,
    key="power_input"
)

# --------------------------------------------------
# ENCODE USER INPUTS (CRITICAL STEP)
# --------------------------------------------------
model_encoded = encoders["model_name"].transform([model_name_input])[0]
location_encoded = encoders["location"].transform([location_input])[0]
owner_encoded = encoders["owner"].transform([owner_input])[0]

#--------------------------------------------------
# FEATURE ENGINEERING & TARGET DEFINITION
#--------------------------------------------------
# Target variable (price)
data['log_price'] = np.log1p(data['price'])

# Features used for prediction
X = data    [
    [
        'model_year',
        'kms_driven',
        'mileage',
        'power',
        'model_name',
        'location',
        'owner'
    ]
]

# Target
y = data['log_price']

# --------------------------------------------------
# MODEL TRAINING FUNCTION
# --------------------------------------------------
@st.cache_resource
def train_model(X, y):
    lr = LinearRegression()
    lr.fit(X, y)
    return lr

model = train_model(X, y)

# --------------------------------------------------
# BUILD INPUT ARRAY (MATCH TRAINING ORDER)
# --------------------------------------------------
input_data = np.array([[
    model_year := 2024 - bike_age,
    kms_driven,
    mileage,
    power,
    model_encoded,
    location_encoded,
    owner_encoded
]])

# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Summary")
    st.write(f"**Model:** {model_name_input}")
    st.write(f"**Location:** {location_input}")
    st.write(f"**Owner Type:** {owner_input}")
    st.write(f"**Bike Age:** {bike_age} years")
    st.write(f"**Kilometers Driven:** {kms_driven:,} km")
    st.write(f"**Mileage:** {mileage} kmpl")
    st.write(f"**Power:** {power} bhp")

with col2:
    st.subheader("💰 Price Prediction")

    if st.button("Predict Price"):
        log_price_pred = model.predict(input_data)[0]

        price_pred = np.expm1(log_price_pred)

        st.success(
            f"Estimated Price:\n₹ {int(price_pred):,}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#ff7a18;">
        Built by Anoushka Thakur • Data Analyst Portfolio Project
    </p>
    """,
    unsafe_allow_html=True
)
