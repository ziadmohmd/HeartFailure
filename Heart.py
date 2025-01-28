#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

filename = 'RandomForestClassifier_Heart.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Heart Falilure Prediction App')
st.subheader('Please enter your data:')

df = pd.read_csv('/content/heart.csv')
columns_list = df.columns.to_list()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    object_columns = df.select_dtypes(include=['object']).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    df_numerical = pd.DataFrame(numerical_transformer.fit_transform(df[numerical_columns]), columns=numerical_columns)
    df_categorical = pd.get_dummies(df[object_columns], columns=object_columns)
    
    df_preprocessed = pd.concat([df_numerical, df_categorical], axis=1)
    
    df_preprocessed = df_preprocessed.reindex(columns=columns_list, fill_value=0)

    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.subheader('Heart Failure:')
    st.write(prediction_text)