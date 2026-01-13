# cnn_lstm_models.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, TimeDistributed, LSTM, Bidirectional,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
)

def build_base_cnn(input_shape):
    cnn_input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    return Model(cnn_input, x, name="base_cnn")

def build_model(model_type="A2"):
    sequence_input = Input(shape=(10, 32, 32, 10))  # (T, H, W, C)
    cnn_model = build_base_cnn((32, 32, 10))
    x = TimeDistributed(cnn_model)(sequence_input)  # (B, T, F)

    if model_type == "A1":  # CNN + LSTM
        x = LSTM(64)(x)

    elif model_type == "A2":  # CNN + BiLSTM
        x = Bidirectional(LSTM(64))(x)

    elif model_type == "A4":  # CNN + BiLSTM + MultiHeadAttention
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(x)

    elif model_type == "A5":  # CNN + Transformer Encoder
        # Positional embedding
        pos = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_embed = tf.keras.layers.Embedding(input_dim=100, output_dim=x.shape[-1])(pos)
        x += pos_embed

        # Transformer encoder block
        attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)

        ffn = Dense(128, activation='relu')(x)
        ffn = Dense(x.shape[-1])(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)

        x = GlobalAveragePooling1D()(x)

    elif model_type == "A6":  # CNN-only
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    elif model_type == "A7":  # CNN + Multi-Head Attention (No LSTM)
        # Shape: (B, T, F) after TimeDistributed CNN
        attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(attn_out)

    elif model_type == "A8":  # CNN + LSTM + Multi-Head Attention
        x = LSTM(64, return_sequences=True)(x)
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(x)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
        

    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, output, name=f"model_{model_type}")
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
    return model
