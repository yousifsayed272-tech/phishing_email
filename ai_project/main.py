# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow import keras
from tensorflow.keras import layers

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("phishing_email.csv")

df = df[['text_combined', 'label']]
df.columns = ['text', 'label']
df.dropna(inplace=True)

print("Dataset Loaded")

# ==============================
# 3. Split Data
# ==============================
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Vectorization
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# حفظ vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ==============================
# 5. Build Model
# ==============================
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_vec.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==============================
# 6. Train
# ==============================
model.fit(
    X_train_vec, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_vec, y_test)
)

# ==============================
# 7. Save Model
# ==============================
model.save("phishing_model.h5")

# ==============================
# 8. Evaluate
# ==============================
loss, acc = model.evaluate(X_test_vec, y_test)
print("Final Accuracy:", acc)