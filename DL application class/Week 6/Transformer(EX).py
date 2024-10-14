import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras  # keras 임포트 추가
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def readucr(filename):
    # CSV 파일을 pandas로 읽기 (header 포함)
    data = pd.read_csv(filename)

    # 마지막 두 열을 문자열로 처리
    string_columns = data.iloc[:, -2:].astype(str)
    # 나머지 열을 float으로 변환
    float_columns = data.iloc[:, :-2].astype(float)

    # 최종 데이터셋 구성
    x = float_columns
    y = string_columns.iloc[:, -1]  # 마지막 열을 라벨로 사용 (문자열)

    return x, y

# root_dir에 지정된 파일에서 데이터를 읽음
root_dir = "/Users/hwang-gyuhan/Desktop/Collage/3-2/딥러닝응용/4주차/DSA_features.csv"

x, y = readucr(root_dir)

# y가 범주형(문자열)인 경우 숫자로 변환 (라벨 인코딩)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # y를 숫자형으로 변환

# 데이터 섞기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Conv1D 모델을 사용하기 위해 입력 데이터를 3차원으로 reshape
x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)  # float32로 변환
x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)    # float32로 변환

# 클래스 수 확인
n_classes = len(np.unique(y_train))

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)  # keras를 사용하여 Input 정의
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)
