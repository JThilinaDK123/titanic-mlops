import streamlit as st
import pandas as pd
import seaborn as sns
from model import train_model, predict_survival

st.title("🚢 Titanic Survival Predictor")

# Load Titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Train a simple model
model, features = train_model(df)

st.write("### Try it out!")

# Simple input form
pclass = st.selectbox("Passenger Class (1=First, 2=Second, 3=Third)", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=32.2)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "fare": fare,
        "embarked": embarked
    }])
    prediction = predict_survival(model, input_data, features)
    st.success("✅ Survived!" if prediction == 1 else "❌ Did not survive")
