"""
Model Eğitim Scripti
Türkçe Haber Kategorisi Sınıflandırma Projesi
"""

import pandas as pd
import numpy as np
import re
import pickle
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, 
    Bidirectional, Conv1D, GlobalMaxPooling1D, 
    Concatenate, SpatialDropout1D, BatchNormalization
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import stopwords

# Parametreler
MAX_WORDS = 15000
MAX_LEN = 250
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 25

# Türkçe stopwords
nltk.download('stopwords', quiet=True)
turkish_stopwords = set(stopwords.words('turkish'))
extra_stopwords = {
    'dedi', 'söyledi', 'açıkladı', 'belirtti', 'kaydetti', 'etti', 'olan',
    'üzere', 'göre', 'karşı', 'önce', 'sonra', 'bugün', 'dün', 'yarın'
}
turkish_stopwords.update(extra_stopwords)


def preprocess_text(text):
    """Metin ön işleme fonksiyonu"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-züğışöçıİ\s]', ' ', text)
    text = ' '.join(text.split())
    
    words = text.split()
    words = [w for w in words if w not in turkish_stopwords and len(w) > 2]
    
    return ' '.join(words)


def load_and_prepare_data(data_path):
    """Veri setini yükle ve hazırla"""
    print("Veri seti yükleniyor...")
    df = pd.read_csv(data_path)
    
    # "Genel" kategorisini çıkar
    df = df[df['category'] != 'genel']
    
    print(f"Toplam örnek sayısı: {len(df)}")
    print("\nKategori dağılımı:")
    print(df['category'].value_counts())
    
    # Metin temizleme
    print("\nMetinler temizleniyor...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    df = df[df['text_clean'].str.len() > 20]
    
    return df


def create_sequences(df):
    """Tokenizasyon ve sequence oluşturma"""
    print("\nTokenizasyon yapılıyor...")
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['text_clean'])
    
    sequences = tokenizer.texts_to_sequences(df['text_clean'])
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    print(f"Kelime dağarcığı boyutu: {len(tokenizer.word_index)}")
    print(f"Sınıf sayısı: {len(label_encoder.classes_)}")
    print(f"Sınıflar: {label_encoder.classes_}")
    
    return X, y, tokenizer, label_encoder


def split_data(X, y):
    """Veriyi eğitim, validasyon ve test olarak böl"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print("\nVeri bölümü:")
    print(f"Eğitim: {len(X_train)} örnek")
    print(f"Validasyon: {len(X_val)} örnek")
    print(f"Test: {len(X_test)} örnek")
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict


def build_hybrid_model(num_classes):
    """Hybrid LSTM + CNN modelini oluştur"""
    print("\nModel oluşturuluyor...")
    
    input_layer = Input(shape=(MAX_LEN,))
    
    embedding = Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN
    )(input_layer)
    
    embedding = SpatialDropout1D(0.3)(embedding)
    
    # LSTM Branch
    lstm_branch = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    lstm_branch = Dropout(0.3)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(64))(lstm_branch)
    
    # CNN Branch 1
    cnn_branch = Conv1D(128, 3, activation='relu', padding='same')(embedding)
    cnn_branch = GlobalMaxPooling1D()(cnn_branch)
    
    # CNN Branch 2
    cnn_branch2 = Conv1D(128, 5, activation='relu', padding='same')(embedding)
    cnn_branch2 = GlobalMaxPooling1D()(cnn_branch2)
    
    # Merge branches
    merged = Concatenate()([lstm_branch, cnn_branch, cnn_branch2])
    
    # Dense layers
    dense = Dense(128, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(0.3)(dense)
    
    output = Dense(num_classes, activation='softmax')(dense)
    
    model = Model(inputs=input_layer, outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, class_weight_dict):
    """Modeli eğit"""
    print("\nModel eğitimi başlıyor...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, X_val, y_val, label_encoder):
    """Modeli değerlendir"""
    print("\nModel değerlendirmesi yapılıyor...")
    
    # Test predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Validation predictions
    val_pred = model.predict(X_val, verbose=0)
    val_pred_classes = np.argmax(val_pred, axis=1)
    
    test_accuracy = accuracy_score(y_test, y_pred_classes)
    val_accuracy = accuracy_score(y_val, val_pred_classes)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred_classes, 
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    print("\nKategori bazında doğruluk:")
    for i, cat in enumerate(label_encoder.classes_):
        mask = y_test == i
        cat_accuracy = accuracy_score(y_test[mask], y_pred_classes[mask])
        print(f"{cat:10}: {cat_accuracy*100:.2f}%")
    
    return y_pred_classes, test_accuracy, val_accuracy


def plot_confusion_matrix(y_test, y_pred_classes, label_encoder, test_accuracy):
    """Confusion matrix grafiğini çiz"""
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={'label': 'Sample Count'}
    )
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy*100:.2f}%', 
              fontsize=16, weight='bold', pad=20)
    plt.ylabel('True Category', fontsize=12)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix kaydedildi: results/confusion_matrix.png")


def plot_training_history(history):
    """Eğitim grafiklerini çiz"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy plot
    ax1.plot(epochs_range, history.history['accuracy'], 'b-', 
             label='Training Accuracy', linewidth=2)
    ax1.plot(epochs_range, history.history['val_accuracy'], 'r-', 
             label='Validation Accuracy', linewidth=2)
    ax1.axhline(y=0.90, color='g', linestyle='--', alpha=0.5, label='90% Target')
    ax1.set_title('Model Accuracy', fontsize=14, weight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs_range, history.history['loss'], 'b-', 
             label='Training Loss', linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], 'r-', 
             label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, weight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    print("Eğitim grafikleri kaydedildi: results/training_history.png")


def save_artifacts(model, tokenizer, label_encoder):
    """Model ve diğer objeleri kaydet"""
    print("\nModel ve artifacts kaydediliyor...")
    
    model.save('models/news_classifier_hybrid.keras')
    print("Model kaydedildi: models/news_classifier_hybrid.keras")
    
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer kaydedildi: models/tokenizer.pkl")
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder kaydedildi: models/label_encoder.pkl")


def main():
    parser = argparse.ArgumentParser(description='Train news classification model')
    parser.add_argument('--data', type=str, default='data/haberler_top5.csv',
                        help='Path to dataset CSV file')
    args = parser.parse_args()
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load and prepare data
    df = load_and_prepare_data(args.data)
    
    # Create sequences
    X, y, tokenizer, label_encoder = create_sequences(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict = split_data(X, y)
    
    # Build model
    num_classes = len(label_encoder.classes_)
    model = build_hybrid_model(num_classes)
    model.summary()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val, class_weight_dict)
    
    # Evaluate model
    y_pred_classes, test_accuracy, val_accuracy = evaluate_model(
        model, X_test, y_test, X_val, y_val, label_encoder
    )
    
    # Plot results
    plot_confusion_matrix(y_test, y_pred_classes, label_encoder, test_accuracy)
    plot_training_history(history)
    
    # Save artifacts
    save_artifacts(model, tokenizer, label_encoder)
    
    print("\nEğitim tamamlandı!")
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracy*100:.2f}%")


if __name__ == '__main__':
    main()