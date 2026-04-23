import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# mengload data
df = pd.read_csv("titanic_preprocessing.csv")

# memisahkan antara fitur dan target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# mengsplit data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# menampilkan akurasi hasil training model
acc = model.score(X_test, y_test)
print("Accuracy:", acc)

# menyimpan model
os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.pkl")
print("Model disimpan di outputs/")