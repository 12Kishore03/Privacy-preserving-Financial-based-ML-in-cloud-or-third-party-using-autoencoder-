import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# paths
DATA_PATH = "../data/raw/dataset.csv"
MODEL_DIR = "../server/model"
ENCODED_OUT = "../data/encoded/encoded_data.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("../data/encoded", exist_ok=True)

# load data
df = pd.read_csv(DATA_PATH)

# -----------------------------
# drop identifiers & text-heavy columns
# -----------------------------
DROP_COLS = [
    "Customer_ID", "Customer_Name", "Customer_Email", "Customer_Contact",
    "Transaction_ID", "Transaction_Description",
    "State", "City", "Bank_Branch", "Transaction_Location", "Merchant_ID"
]

df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# -----------------------------
# extract label
# -----------------------------
y = df["Is_Fraud"]
df.drop(columns=["Is_Fraud"], inplace=True)

# -----------------------------
# time feature engineering
# -----------------------------
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
df["Transaction_Time"] = pd.to_datetime(df["Transaction_Time"], format="%H:%M:%S")

df["day_of_week"] = df["Transaction_Date"].dt.weekday
df["hour_of_day"] = df["Transaction_Time"].dt.hour

df.drop(columns=["Transaction_Date", "Transaction_Time"], inplace=True)

# -----------------------------
# categorical encoding
# -----------------------------
CATEGORICAL_COLS = [
    "Gender", "Account_Type", "Transaction_Type",
    "Merchant_Category", "Transaction_Device",
    "Device_Type", "Transaction_Currency"
]

encoders = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -----------------------------
# scale numeric features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------
# PCA encoder
# -----------------------------
pca = PCA(n_components=0.95)
X_encoded = pca.fit_transform(X_scaled)

# -----------------------------
# save outputs
# -----------------------------
encoded_df = pd.DataFrame(X_encoded)
encoded_df["Is_Fraud"] = y.values
encoded_df.to_csv(ENCODED_OUT, index=False)

joblib.dump(pca, f"{MODEL_DIR}/encoder.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/cat_encoders.pkl")

print("PCA encoder trained")
print("Original features:", df.shape[1])
print("Encoded features :", X_encoded.shape[1])


