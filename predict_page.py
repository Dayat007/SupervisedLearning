import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import sklearn.externals
import joblib

def show_predict_page():
    st.title("Software FinishMill Prediction")

    st.write("We need some information to predict the Quality")

    Total_Feed_Actual = st.slider("Total Feed Actual (TPH)", 0.0, 200.0, 181.186734)

    Gyps_Feed_Actual = st.slider("Gyps Feed Actual (TPH)", 0.0, 20.0, 6.280761)

    LS_Feed_Actual = st.slider("LS Feed Actual (TPH)", 0.0, 20.0, 13.311128)

    Clinker_Feed_Actual = st.slider("Clinker Feed Actual (TPH)", 0.0, 200.0, 161.594844)

    Ok = st.button("Prediction for Classifier")
    if Ok:
        model = joblib.load('model-finishmill.joblib')
        new_data = [Total_Feed_Actual,Gyps_Feed_Actual,LS_Feed_Actual,Clinker_Feed_Actual]
        prediction = model.predict([new_data])
        st.subheader(prediction)

    Ok2 = st.button("Prediction for Regresi")
    if Ok2:
        model = joblib.load('model-finishmill-regresi.joblib')
        new_data = [Total_Feed_Actual,Gyps_Feed_Actual,LS_Feed_Actual,Clinker_Feed_Actual]
        prediction = model.predict([new_data])
        st.subheader(prediction)
