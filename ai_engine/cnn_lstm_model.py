import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    LSTM,
    Dense,
    Input,
    concatenate,
    BatchNormalization,
    Dropout,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def create_cnn_lstm_model(input_shape=(100, 5, 4)):
    # Input layers for each timeframe
    inputs = [
        Input(shape=(input_shape[0], input_shape[1])) for _ in range(input_shape[2])
    ]
    # Parallel CNN branches
    conv_branches = []
    for i in range(input_shape[2]):
        x = Conv1D(64, 5, activation="relu", padding="same")(inputs[i])
        x = BatchNormalization()(x)
        x = Conv1D(128, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        conv_branches.append(x)
    # Concatenate and LSTM
    merged = concatenate(conv_branches)
    reshaped = Reshape((input_shape[0], -1))(merged)
    lstm_out = LSTM(256, return_sequences=True)(reshaped)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = LSTM(128)(lstm_out)
    # Residual connections
    residual = Dense(128)(lstm_out)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(residual)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x + residual)
    model = Model(inputs=inputs, outputs=out)
    # Metal-optimized compilation
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )
    return model
