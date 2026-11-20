from pathlib import Path

import pandas as pd
import tensorflow as tf  # Thư viện xây dựng mạng nơ-ron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Đọc dữ liệu đã được chia sẵn
thu_muc = Path(__file__).resolve().parent
train_data = pd.read_csv(thu_muc/ "train.csv")
test_data = pd.read_csv(thu_muc / "test.csv")

# Tách thuộc tính đầu vào và nhãn đầu ra
x_train = train_data.drop(columns = ["Outcome"])
y_train = train_data["Outcome"]

# Lấy các cột đầu vào theo đúng thứ tự
x_test = test_data[x_train.columns]
y_test = test_data["Outcome"]

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train).astype("float32") # Tính trung bình và độ lệch chuẩn cho từng cột trên tập train
x_test = scaler.transform(x_test).astype("float32")

# Khai báo đầu vào và xây dựng AMNN
tf.keras.utils.set_random_seed(42) # .keras dùng để xây dựng và huấn luyện mạng nơ-ron 
layers = tf.keras.layers

dau_vao = tf.keras.Input(shape = (x_train.shape[1],), name = "dau_vao")

# Nhánh cộng: 16 nơ-ron học các tổng có trọng số
nhanh_cong = layers.Dense(16, activation= "relu", name = "nhanh_cong")(dau_vao)

# Nhánh nhân: nhân các đặc trưng của từng người với nhau
nhanh_nhan = layers.Lambda(lambda x: tf.reduce_prod(x, axis = 1, keepdims = True), output_shape = (1,), name = "nhanh_nhan")(dau_vao)

# Cộng giá trị của nhánh nhân vào từng đầu ra của nhánh cộng
ket_hop = layers.Add(name = "ket_hop")([nhanh_cong, nhanh_nhan])
lop_an = layers.Dense(8, activation = "relu", name = "lop_an")(ket_hop)
dau_ra = layers.Dense(1, activation = "sigmoid", name = "xac_suat")(lop_an)

# Tạo và huấn luyện mô hình
model = tf.keras.Model(inputs = dau_vao, outputs = dau_ra, name = "AMNN")

# Thiết lập cách huấn luyện
model.compile(optimizer = "adam", loss = "binary_crossentropy")

# Huấn luyên trên train.csv
model.fit(x_train, y_train, epochs = 100, batch_size = 32, verbose = 2)

# Dự đoán trên test.csv
y_prob = model.predict(x_test, verbose = 0).ravel() # yêu cầu mô hình đã học đưa ra dự đoán
y_pred = (y_prob >= 0.5).astype(int)

print("\n KẾT QUẢ MÔ HÌNH AMNN")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division = 0))
print("Recall:", recall_score(y_test, y_pred, zero_division = 0))
print("F1-score:", f1_score(y_test, y_pred, zero_division = 0))

print("\n Confusion Matrrix:")
print(confusion_matrix(y_test, y_pred, labels = [0, 1]))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, digits = 4, zero_division = 0))