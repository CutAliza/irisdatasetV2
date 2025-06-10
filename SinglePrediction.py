import streamlit as st
import joblib
import numpy as np

#model = joblib.load("modelJb_klasifikasiIris.joblib")

def show_single():
    st.title("Single Prediction")
    a=float(st.number_input("Sepal length in cm"))
    b=float(st.number_input("Sepal width in cm"))
    c=float(st.number_input("Petal length in cm"))
    d=float(st.number_input("Petal width in cm"))

    # Checkbox untuk memilih model
    use_knn = st.checkbox("Use KNN")
    use_svm = st.checkbox("Use SVM")
    use_nn = st.checkbox("Use Neural Network")
    use_dt = st.checkbox("Use Decision Tree")

    btn = st.button("Predict")

    if btn:
        input_data = np.array([a, b, c, d]).reshape(1, -1)

        def show_prediction(model_name, model_file):
            model = joblib.load(model_file)
            pred = model.predict(input_data)
            label = ""
            if pred[0] == 0:
                label = "Iris-setosa"
            elif pred[0] == 1:
                label = "Iris-versicolor"
            elif pred[0] == 2:
                label = "Iris-virginica"
            else:
                label = "Unknown"
            st.subheader(f"{model_name} Prediction: {pred[0]} → {label}")

        if use_knn:
            show_prediction("K-Nearest Neighbors", "modelJb_KNN.joblib")
        if use_svm:
            show_prediction("Support Vector Machine", "modelJb_SVM.joblib")
        if use_nn:
            show_prediction("Neural Network", "modelJb_NN.joblib")
        if use_dt:
            show_prediction("Decision Tree", "modelJb_DecisionTree.joblib")