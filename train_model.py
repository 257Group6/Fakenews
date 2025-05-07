import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Load and Label Dataset ===
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df])
df = df[["text", "label"]].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === 2. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# === 3. Tokenization and Padding ===
MAX_WORDS = 10000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

# === 4. Build the Model (with input_shape) ===
model = Sequential()
model.add(Embedding(input_dim=MAX_WORDS, output_dim=64, input_shape=(MAX_LEN,)))  # ✅ FIXED: input_shape
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# === 5. Train the Model ===
es = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[es])

# === 6. Evaluate the Model ===
loss, acc = model.evaluate(X_test_pad, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# === 7. Classification Report ===
y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# === 8. Save Model & Tokenizer ===
model.save("fake_news_model.keras")  # ✅ native format
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# === 9. Plot Accuracy ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
