import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("lc_march25.csv")

# Prepare features and target
features = data.drop(["LUNG_CANCER"], axis=1)
target = data["LUNG_CANCER"].map({'YES': 1, 'NO': 0})  # Convert to numerical

# One-hot encoding for categorical variables
nfeatures = pd.get_dummies(features)
feature_columns = nfeatures.columns

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
sfeatures = scaler.fit_transform(nfeatures)

# Split data for training
df_train, df_test, target_train, target_test = train_test_split(sfeatures, target, test_size=0.2, random_state=42)

# Train KNN Model
k = int(len(data) ** 0.5)
k = k + 1 if k % 2 == 0 else k  # Ensure k is odd
model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
model.fit(df_train, target_train)

# Streamlit UI Enhancements
st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f4f4f4;}
    .stRadio label {font-size: 16px !important;}
    .stButton>button {background-color: #4CAF50; color: white; padding: 10px 24px; font-size: 16px; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Lung Cancer Prediction App 🚀</h1>", unsafe_allow_html=True)
st.write("### Enter details below to check the risk of lung cancer.")

# User Inputs
gender = st.radio("Select Gender", ["Male", "Female"])
age = st.slider("Select Age", 30, 100, 50)

# Improving radio buttons with "Yes" or "No" instead of numbers
smoking = st.radio("Do you smoke?", ["No", "Yes"])
yellow_fingers = st.radio("Do you have yellow fingers?", ["No", "Yes"])
anxiety = st.radio("Do you experience anxiety?", ["No", "Yes"])
peer_pressure = st.radio("Are you influenced by peer pressure?", ["No", "Yes"])
chronic_disease = st.radio("Do you have any chronic disease?", ["No", "Yes"])
fatigue = st.radio("Do you feel fatigued often?", ["No", "Yes"])
allergy = st.radio("Do you have allergies?", ["No", "Yes"])
wheezing = st.radio("Do you experience wheezing?", ["No", "Yes"])
alcohol = st.radio("Do you consume alcohol frequently?", ["No", "Yes"])
coughing = st.radio("Do you have persistent coughing?", ["No", "Yes"])
shortness_of_breath = st.radio("Do you have shortness of breath?", ["No", "Yes"])
swallowing_difficulty = st.radio("Do you have difficulty swallowing?", ["No", "Yes"])
chest_pain = st.radio("Do you experience chest pain?", ["No", "Yes"])

# Convert categorical variables to one-hot encoding
gender_m = 1 if gender == "Male" else 0
gender_f = 1 if gender == "Female" else 0

# Convert Yes/No responses to numerical format
def yes_no_to_numeric(response):
    return 1 if response == "Yes" else 0

input_dict = {
    "AGE": age,
    "SMOKING": yes_no_to_numeric(smoking),
    "YELLOW_FINGERS": yes_no_to_numeric(yellow_fingers),
    "ANXIETY": yes_no_to_numeric(anxiety),
    "PEER_PRESSURE": yes_no_to_numeric(peer_pressure),
    "CHRONIC_DISEASE": yes_no_to_numeric(chronic_disease),
    "FATIGUE": yes_no_to_numeric(fatigue),
    "ALLERGY": yes_no_to_numeric(allergy),
    "WHEEZING": yes_no_to_numeric(wheezing),
    "ALCOHOL_CONSUMPTION": yes_no_to_numeric(alcohol),
    "COUGHING": yes_no_to_numeric(coughing),
    "SHORTNESS_OF_BREATH": yes_no_to_numeric(shortness_of_breath),
    "SWALLOWING_DIFFICULTY": yes_no_to_numeric(swallowing_difficulty),
    "CHEST_PAIN": yes_no_to_numeric(chest_pain),
    "GENDER_F": gender_f,
    "GENDER_M": gender_m
}

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Apply MinMaxScaler on input data
input_data_scaled = scaler.transform(input_data)

# Prediction & Explanation
if st.button("Predict Lung Cancer Risk"):
    prediction = model.predict(input_data_scaled)[0]

    if prediction == 1:
        # High Risk UI
        st.markdown("<h2 style='color: red;'>⚠️ High Risk of Lung Cancer ⚠️</h2>", unsafe_allow_html=True)
        st.error("Your responses indicate a **high risk of lung cancer**. Please consult a doctor for further evaluation.")
        
        # Advice
        st.markdown("### 🚑 Health Advice:")
        st.markdown("""
        - 🚭 **Quit Smoking:** Smoking is the leading cause of lung cancer. Seek professional help if needed.
        - 🍎 **Healthy Diet:** Increase intake of vegetables, fruits, and whole grains.
        - 🏃 **Exercise Regularly:** Helps in improving lung function and overall health.
        - 🏥 **Medical Check-ups:** Regular screenings can help detect issues early.
        - 🌱 **Avoid Pollutants:** Reduce exposure to harmful chemicals and secondhand smoke.
        """)
    
    else:
        # Low Risk UI
        st.markdown("<h2 style='color: green;'>✅ Low Risk of Lung Cancer ✅</h2>", unsafe_allow_html=True)
        st.success("Good news! You have a **low risk of lung cancer**. However, continue maintaining a healthy lifestyle.")

        # Confetti Effect 🎉
        st.balloons()

        # Preventive Measures
        st.markdown("### 🌿 Preventive Measures:")
        st.markdown("""
        - ✅ **Continue Healthy Habits:** Avoid smoking and limit alcohol consumption.
        - 🍵 **Stay Hydrated:** Drinking enough water helps in keeping lungs clean.
        - 🌿 **Breathe Fresh Air:** Practice deep breathing exercises and spend time in nature.
        - 🏋️ **Stay Active:** Keep a regular fitness routine to maintain lung health.
        """)

