import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Load models
model_files = [f for f in os.listdir() if f.endswith('.joblib')]
models = {f.split('_')[0]: joblib.load(f) for f in model_files}

st.title('High/Low Prediction Dashboard')

# Pilihan model yang konsisten dan user-friendly
def load_model(model_key):
    mapping = {
        'KNN': 'modelJb_HighLowSpenderKNN.joblib',
        'SVM': 'modelJb_HighLowSpenderSVM.joblib',
        'DT': 'modelJb_HighLowSpenderDesicionTree.joblib',
        'NN': 'modelJb_HighLowSpenderNN.joblib',
    }
    return joblib.load(mapping[model_key])

model_options = ['KNN', 'SVM', 'DT', 'NN']

menu = st.sidebar.radio('Pilih Mode:', [
    'Prediksi Satuan',
    'Prediksi Batch',
    'Prediksi Satuan (Regresi)',
    'Prediksi Batch (Regresi)'
])

# Fungsi prediksi
@st.cache_data
def predict_single(model_name, payment_type_encoded, sequential, installments, value):
    X = np.array([[payment_type_encoded, sequential, installments, value]])
    pred = models[model_name].predict(X)[0]
    return 'High Spender' if pred == 1 else 'Low Spender'

# Fungsi untuk load scaler
@st.cache_resource
def load_scaler():
    # Pastikan scaler yang digunakan sama dengan saat training
    return joblib.load('scaler_HighLowSpenderSVM.joblib')

# Daftar fitur sesuai training model (letakkan di atas, sebelum if menu)
fitur_regresi_svm = [
    'Certificates', 'Years of Experience', 'age', 'Time Arrival Strafe',
    'Project Proximity', 'Violation Risk Index', 'Company PCAB Score',
    'Weekly Overtime Hours', 'Salary Bracket', 'is_good'
]
fitur_regresi_10 = [
    'Certificates', 'Years of Experience', 'age', 'Time Arrival Strafe',
    'Project Cost', 'Project Proximity', 'Violation Risk Index',
    'Company PCAB Score', 'Weekly Overtime Hours', 'Salary Bracket'
]
fitur_regresi_9 = [
    'Certificates', 'Years of Experience', 'age', 'Time Arrival Strafe',
    'Project Cost', 'Project Proximity', 'Violation Risk Index',
    'Company PCAB Score', 'Weekly Overtime Hours'
]
fitur_regresi_nn = [
    'Certificates', 'Years of Experience', 'age', 'Time Arrival Strafe',
    'Project Proximity', 'Violation Risk Index', 'Company PCAB Score',
    'Weekly Overtime Hours', 'Salary Bracket'
]

if menu == 'Prediksi Satuan':
    st.header('Prediksi Satuan')
    model_klasifikasi = st.selectbox(
        'Pilih Model',
        model_options,
        format_func=lambda x: x.upper()
    )
    sequential = st.number_input('Payment Sequential', min_value=1, value=1)
    installments = st.number_input('Payment Installments', min_value=1, value=1)
    value = st.number_input('Payment Value', min_value=0.0, value=0.0, step=100.0)
    if st.button('Prediksi'):
        model = load_model(model_klasifikasi)
        X = np.array([[sequential, installments, value]])
        pred = model.predict(X)[0]
        result = 'High Spender' if pred == 1 else 'Low Spender'
        st.success(f'Prediksi: {result}')

elif menu == 'Prediksi Batch':
    st.header('Prediksi Batch')
    # Pilihan model batch
    batch_model_options = {
        'High/Low Spender (SVM)': {
            'file': 'modelJb_HighLowSpenderSVM.joblib',
            'scaler': 'scaler_HighLowSpenderSVM.joblib',
            'required_cols': ['payment_sequential', 'payment_installments', 'payment_value'],
            'type': 'klasifikasi',
        },
        'High/Low Spender (Decision Tree)': {
            'file': 'modelJb_HighLowSpenderDesicionTree.joblib',
            'scaler': None,
            'required_cols': ['payment_sequential', 'payment_installments', 'payment_value'],
            'type': 'klasifikasi',
        },
        'High/Low Spender (NN)': {
            'file': 'modelJb_HighLowSpenderNN.joblib',
            'scaler': None,
            'required_cols': ['payment_sequential', 'payment_installments', 'payment_value'],
            'type': 'klasifikasi',
        },
        'High/Low Spender (KNN)': {
            'file': 'modelJb_HighLowSpenderKNN.joblib',
            'scaler': None,
            'required_cols': ['payment_sequential', 'payment_installments', 'payment_value'],
             'type': 'klasifikasi',
     },
    }
    batch_model_name = st.selectbox('Pilih Model Batch', list(batch_model_options.keys()))
    uploaded_file = st.file_uploader('Upload file CSV', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Data yang diupload:', df.head())
        config = batch_model_options[batch_model_name]
        required_cols = config['required_cols']
        if all(col in df.columns for col in required_cols):
            df = df.dropna(subset=required_cols)
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            X = df[required_cols].values
            if config['scaler']:
                scaler = joblib.load(config['scaler'])
                X = scaler.transform(X)
            model = joblib.load(config['file'])
            # Cek apakah model hasil load benar-benar objek model
            if not hasattr(model, 'predict'):
                st.error(f"File model '{config['file']}' bukan objek model yang valid. Pastikan file tersebut hasil training model sklearn.")
                st.stop()
            preds = model.predict(X)
            if config['type'] == 'klasifikasi':
                df['Prediksi'] = ['High Spender' if p == 1 else 'Low Spender' for p in preds]
            else:
                df['Prediksi'] = preds
            st.write('Hasil Prediksi:', df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Hasil', csv, 'hasil_prediksi.csv', 'text/csv')
        else:
            st.error(f'Kolom pada file harus: {required_cols}')

elif menu == 'Prediksi Satuan (Regresi)':
    st.header('Prediksi Satuan (Regresi)')
    model_regresi_options = {
        'SVM Regresi': 'modelJb_HighLowSpenderSVMRegresi.joblib',
        'Decision Tree Regresi': 'modelJb_HighLowSpenderDesicionTreeRegresi.joblib',
        'NN Regresi': 'modelJb_HighLowSpenderNNRegresi.joblib',
        'KNN Regresi': 'modelJb_HighLowSpenderKNNRegresi.joblib',
    }
    model_regresi = st.selectbox('Pilih Model Regresi', list(model_regresi_options.keys()))
    if model_regresi == 'NN Regresi':
        fitur_regresi = fitur_regresi_nn
    elif model_regresi == 'SVM Regresi':
        fitur_regresi = fitur_regresi_svm
    else:
        fitur_regresi = fitur_regresi_10
    input_fitur = []
    for f in fitur_regresi:
        val = st.number_input(f, key=f)
        input_fitur.append(val)
    if st.button('Prediksi Regresi'):
        model = joblib.load(model_regresi_options[model_regresi])
        X_df = pd.DataFrame([input_fitur], columns=fitur_regresi)
        # Gunakan .values jika model tidak punya feature_names_in_
        if (model_regresi == 'Decision Tree Regresi' or model_regresi == 'KNN Regresi') and not hasattr(model, 'feature_names_in_'):
            pred = model.predict(X_df.values)[0]
        else:
            pred = model.predict(X_df)[0]
        threshold = 500000
        label = 1 if pred >= threshold else 0
        st.success(f'Hasil Prediksi (Regresi): {pred} → {label} (1=High, 0=Low)')
        # model = joblib.load(model_regresi_options[model_regresi])
        # X_df = pd.DataFrame([input_fitur], columns=fitur_regresi)
        # # Validasi nama kolom
        # if hasattr(model, 'feature_names_in_'):
        #     fitur_model = list(model.feature_names_in_)
        #     if fitur_model != fitur_regresi:
        #         st.error(f"Urutan/nama fitur input tidak cocok dengan model: {fitur_model}")
        #     else:
        #         pred = model.predict(X_df)[0]
        #         threshold = 500000
        #         label = 1 if pred >= threshold else 0
        #         st.success(f'Hasil Prediksi (Regresi): {pred} → {label} (1=High, 0=Low)')
        # else:
        #     pred = model.predict(X_df)[0]
        #     threshold = 500000
        #     label = 1 if pred >= threshold else 0
        #     st.success(f'Hasil Prediksi (Regresi): {pred} → {label} (1=High, 0=Low)')

elif menu == 'Prediksi Batch (Regresi)':
    st.header('Prediksi Batch (Regresi)')
    batch_regresi_options = {
        'SVM Regresi': 'modelJb_HighLowSpenderSVMRegresi.joblib',
        'NN Regresi': 'modelJb_HighLowSpenderNNRegresi.joblib',
        'Decision Tree Regresi': 'modelJb_HighLowSpenderDesicionTreeRegresi.joblib',
        'KNN Regresi': 'modelJb_HighLowSpenderKNNRegresi.joblib',
    }
    model_regresi = st.selectbox('Pilih Model Batch Regresi', list(batch_regresi_options.keys()))
    uploaded_file = st.file_uploader('Upload file CSV untuk Regresi', type=['csv'], key='regbatch')
    if model_regresi == 'NN Regresi':
        fitur_regresi = fitur_regresi_nn
    elif model_regresi == 'SVM Regresi':
        fitur_regresi = fitur_regresi_svm
    else:
        fitur_regresi = fitur_regresi_10
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Data yang diupload:', df.head())
        required_cols = fitur_regresi
        if all(col in df.columns for col in required_cols):
            df = df.dropna(subset=required_cols)
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=required_cols)
            X_df = df[required_cols]
            model = joblib.load(batch_regresi_options[model_regresi])
            # Gunakan .values jika model tidak punya feature_names_in_
            if (model_regresi == 'Decision Tree Regresi' or model_regresi == 'KNN Regresi') and not hasattr(model, 'feature_names_in_'):
                preds = model.predict(X_df.values)
            else:
                preds = model.predict(X_df)
            threshold = 500000
            labels = [1 if p >= threshold else 0 for p in preds]
            df['Prediksi Regresi'] = preds
            df['High(1)/Low(0)'] = labels
            st.write('Hasil Prediksi Regresi:', df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download Hasil Regresi', csv, 'hasil_prediksi_regresi.csv', 'text/csv')
        else:
            st.error(f'Kolom pada file harus: {required_cols}')
