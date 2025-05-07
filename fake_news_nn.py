import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # Tokenize text
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])

    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=300)

    # Save tokenizer and encoder
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return padded, df['label'], tokenizer, le

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=300),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(data_path="fake_news.csv"):
    X, y, tokenizer, le = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    model.save("fake_news_model.keras")
    print("✅ Model trained and saved as fake_news_model.keras")

if __name__ == "__main__":
    train_model()
