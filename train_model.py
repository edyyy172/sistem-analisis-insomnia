import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================================
# MEMBACA DATASET
# ==========================================================

df = pd.read_excel(
    "dataset sleep health.xlsx",
    sheet_name="Sheet3"
)

print("=" * 50)
print("Jumlah Data :", df.shape)
print("=" * 50)

# ==========================================================
# PREPROCESSING
# ==========================================================

df["Sleep Disorder"] = df["Sleep Disorder"].replace("None", "Normal")
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("Normal")

print("\nDistribusi Label")
print(df["Sleep Disorder"].value_counts())

# ==========================================================
# FITUR
# ==========================================================

X = df[
    [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level"
    ]
]

# ==========================================================
# TARGET
# ==========================================================

label_map = {
    "Normal": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2
}

y = df["Sleep Disorder"].map(label_map)

if y.isnull().sum() > 0:
    print("Masih ada label yang belum dikenali.")
    print(df[y.isnull()]["Sleep Disorder"].unique())
    exit()

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# PIPELINE
# ==========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# ==========================================================
# GRID SEARCH
# ==========================================================

parameter = {
    "knn__n_neighbors": [3,5,7,9,11],
    "knn__metric": ["euclidean","manhattan"],
    "knn__weights": ["uniform","distance"]
}

grid = GridSearchCV(
    pipeline,
    parameter,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

# ==========================================================
# HASIL GRID SEARCH
# ==========================================================

print("\nParameter Terbaik")
print(grid.best_params_)

# ==========================================================
# PREDIKSI
# ==========================================================

prediksi = model.predict(X_test)

akurasi = accuracy_score(y_test, prediksi)

print("\nAkurasi :", round(akurasi,4))

print("\nClassification Report\n")

print(classification_report(y_test, prediksi))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=[
        "Normal",
        "Insomnia",
        "Sleep Apnea"
    ],
    cmap="Blues"
)

plt.title("Confusion Matrix KNN")
plt.show()

# ==========================================================
# SIMPAN MODEL
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/knn_model.pkl"
)

print("\nModel berhasil disimpan di models/knn_model.pkl")