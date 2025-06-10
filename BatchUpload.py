import streamlit as st
import pandas as pd
import numpy as np
import joblib

def get_label(pred):
    return ["Iris-setosa", "Iris-versicolor", "Iris-virginica"][pred] if pred in [0, 1, 2] else "Unknown"

def show_batch():
    st.header("🌸 Batch Flower Prediction")

    uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data", df.head())

        # Pilih model yang akan digunakan
        st.subheader("🧠 Choose Models")
        use_knn = st.checkbox("Use K-Nearest Neighbors")
        use_svm = st.checkbox("Use Support Vector Machine")
        use_nn  = st.checkbox("Use Neural Network")
        use_dt  = st.checkbox("Use Decision Tree")

        if st.button("🔍 Predict"):
            if not any([use_knn, use_svm, use_nn, use_dt]):
                st.warning("Please select at least one model.")
                return

            # Ambil fitur dan pastikan nama kolom sesuai model saat training
            X = df.iloc[:, :4]
            X.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

            if use_knn:
                model = joblib.load("modelJb_KNN.joblib")
                preds = model.predict(X)
                st.markdown("### K-Nearest Neighbors Predictions")
                df_knn = df.copy()
                df_knn["KNN Class"] = preds
                df_knn["KNN Label"] = [get_label(p) for p in preds]
                st.dataframe(df_knn[["KNN Class", "KNN Label"]])

            if use_svm:
                model = joblib.load("modelJb_SVM.joblib")
                preds = model.predict(X)
                st.markdown("### Support Vector Machine Predictions")
                df_svm = df.copy()
                df_svm["SVM Class"] = preds
                df_svm["SVM Label"] = [get_label(p) for p in preds]
                st.dataframe(df_svm[["SVM Class", "SVM Label"]])

            if use_nn:
                model = joblib.load("modelJb_NN.joblib")
                preds = model.predict(X)
                st.markdown("### Neural Network Predictions")
                df_nn = df.copy()
                df_nn["NN Class"] = preds
                df_nn["NN Label"] = [get_label(p) for p in preds]
                st.dataframe(df_nn[["NN Class", "NN Label"]])

            if use_dt:
                model = joblib.load("modelJb_DecisionTree.joblib")
                preds = model.predict(X)
                st.markdown("### Decision Tree Predictions")
                df_dt = df.copy()
                df_dt["DT Class"] = preds
                df_dt["DT Label"] = [get_label(p) for p in preds]
                st.dataframe(df_dt[["DT Class", "DT Label"]])
