import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
# Ganti path jika perlu
DATA_PATH = 'olist_order_payments_dataset.csv'
df = pd.read_csv(DATA_PATH)

# 2. Labeling per transaksi: High Spender jika payment_value > 500.000
THRESHOLD = 500_000
df = df.dropna(subset=['payment_sequential', 'payment_installments', 'payment_value'])
df['label'] = df['payment_value'].apply(lambda x: 1 if x > THRESHOLD else 0)

# 3. Fitur dan target
X = df[['payment_sequential', 'payment_installments', 'payment_value']]
y = df['label']

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 6. Train SVM
clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluasi
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Low Spender', 'High Spender']))

# 8. Simpan model dan scaler
joblib.dump(clf, 'modelJb_HighLowSpenderSVM_pertransaksi.joblib')
joblib.dump(scaler, 'scaler_HighLowSpenderSVM_pertransaksi.joblib')
print('Model dan scaler baru berhasil disimpan.')
