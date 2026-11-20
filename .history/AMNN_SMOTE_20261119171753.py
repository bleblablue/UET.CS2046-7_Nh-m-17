from pathlib import Path

import pandas as pd
import tensorflow as tf

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)

thu_muc = Path(__file__).resolve().parent
train_data = pd.read_csv(thu_muc/ "train.csv")
test_data = pd.read_csv(thu_muc/ "test.csv")

x_train = train_data.drop(columns = ["Outcome"])
y_train = train_data["Outcome"]

x_test = test_data[x_train.columns]
y_test = test_data["Outcome"]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

smote = SMOTE(random_state = 42)
x_train_smote, y_train_smote = smote.fit(x_train_scaled, y_train)

tf.keras.utils.set_random_seed(42)
layers = tf.keras.layers
dau_vao = tf.keras.Input(shape = (x_train_smote.shape[1],), name = "dau_vao")

nhanh_cong = layers.Dense(16, activation = "relu", name = "nhanh_cong")(dau_vao)

nhanh_nhan = layers.Lambda(lambda x: tf.reduce_prod(x, axis = 1, keepdims = True), output_shape = (1,), name = "nhanh_nhan")

ket_hop = layers.Add(name = "ket_hop")([nhanh_cong, nhanh_nhan])
lop_an = layers.Dense(8, activation = "relu", name = "lop_an")(ket_hop)
dau_ra = layers.Dense(1, activation = "sigmoid", name = "dau_ra")(lop_an)

model = tf.keras.Model(inputs = dau_vao, outputs = dau_ra, name = "AMNN")

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x_train_smote, y_train_smote, epochs = 100, batch_size = 32, verbose = 2)

y_prob = model.predict(x_test_scaled, verbose = 0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\n KẾT QUẢ MÔ HÌNH AMNN + SMOTE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division = 0))
print("Recall:", recall_score(y_test, y_pred, zero_division = 0))
print("F1-score:", f1_score(y_test, y_pred, zero_division = 0))

print("\n Confusion Matrrix:")
print(confusion_matrix(y_test, y_pred, labels = [0, 1]))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, digits = 4, zero_division = 0))