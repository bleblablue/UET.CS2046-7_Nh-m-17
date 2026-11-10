#train mô hình lR
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

#đọc dữ liệu đã được chia
train_data=pandas.read_csv("train.csv")
test_data=pandas.read_csv("test.csv")

#tách thuộc tính và nhãn
x_train = train_data.drop(columns=["Outcome"])
y_train = train_data["Outcome"]

x_test = test_data.drop(columns=["Outcome"])
y_test = test_data["Outcome"]
#chuẩn hóa dữ liệu
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#khởi tạo mô hình LR
model=LogisticRegression(max_iter=1)
#train
model.fit(x_train,y_train)
print("training successfully")
#dự đoán
y_pred=model.predict(x_test)
#đánh giá mô hình
print("\n   KET QUA MO HINH LR")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
#ma trận nhầm lẫn: xem cụ thể từng người bị dự đoán sai như nào
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))