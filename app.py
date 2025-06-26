import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib

# Asumsikan kamu sudah punya data 'data_clean' yang bersih dari outlier dan duplikasi
# Ambil hanya kolom fitur sesuai app.py
fitur = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']
X_clean = data_clean[fitur]
y_clean = data_clean['class']  # label target

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_clean)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_clean, test_size=0.2, random_state=42, stratify=y_clean
)

# Latih model Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Simpan model dan scaler dengan nama yang diminta app.py
joblib.dump(model, 'GNB_model.pkl')
joblib.dump(scaler, 'scaler_ecoli.pkl')

print("✅ GNB_model.pkl dan scaler_ecoli.pkl berhasil disimpan.")
