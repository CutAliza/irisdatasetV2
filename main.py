import streamlit as st
from SinglePrediction import show_single
#from BatchUploadPrediction import show_batch
from BatchUpload import show_batch

st.title("🌼 Iris Prediction Dashboard")

menu = st.sidebar.radio("Choose a mode:", ["Single Prediction", "Batch Upload"])
#menu = st.radio("Choose a mode:", ["Single Prediction", "Batch Upload"])

if menu == "Single Prediction":
    show_single()
elif menu == "Batch Upload":
    show_batch()