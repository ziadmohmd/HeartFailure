import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier 

filename = 'RandomForestClassification_Heart.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Heart Failure Prediction')
st.subheader('Please enter your data:')

df = pd.read_csv('features.csv')
columns_list = df.columns.to_list()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    prediction = loaded_model.predict(df)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.subheader('Heart Disease:')
    st.write(prediction_text)