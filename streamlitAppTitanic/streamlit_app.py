# importing the packages
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st   
import matplotlib.pyplot as plt
from utils import PrepProcesor, columns 

# loading data and model
train_data = pd.read_csv("./train.csv")
train_data = train_data.sample(frac=1).reset_index(drop=True)
model = joblib.load('xgboostClassifer.pkl')

st.title('Did they survive? :ship:')
if st.checkbox("Get training data preview"):
    st.table(train_data.dropna().head(10))
    
# PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
passengerid = st.text_input("Input Passenger ID", '843') 
pclass = st.selectbox("Choose class", [1, 2, 3])
name  = st.text_input("Input Passenger Name", 'John Smith')
sex = st.select_slider("Choose sex", ['male','female'])
age = st.slider("Choose age", 0, 100)
sibsp = st.slider("Choose siblings", 0, 10)
parch = st.slider("Choose parch", 0, 2)
ticket = st.text_input("Input Ticket Number", "12345") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.select_slider("Did they Embark?", ['S','C','Q'])

    
def predict(): 
    row = np.array([passengerid, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(X)
    try:
        if prediction[0] == 1: 
            st.success('Passenger Survived :thumbsup:')
        else: 
            st.error('Passenger did not Survive :thumbsdown:') 
            
    except Exception as err:
        print(err)

trigger = st.button('Predict', on_click=predict)

