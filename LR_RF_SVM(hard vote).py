#kết hợp cả ba mô hình rồi bỏ phiếu dự đoán
import pandas
from sklearn.preprocessing import StandardScaler #chuấn hóa dữ liệu cho LR và SVm
from sklearn.pipeline import Pipeline # đặc standar trong từng pineline để mỗi mô hình nhận dữ liệu phù hợp 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier # vote
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

#đọc dữ liệu
train_data=pandas.read_csv("train.csv")
test_data=pandas.read_csv("test.csv")

#tách thuộc tính với nhãn
x_train=train_data.drop(columns=["Outcome"])
y_train=train_data["Outcome"]

x_test=test_data.drop(columns=["Outcome"])
y_test=test_data["Outcome"]

#LR
lr=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000))])

#RF
rf=RandomForestClassifier(n_estimators=100,random_state=42)

#SVM
svm=Pipeline([("scaler",StandardScaler()),("model",SVC(kernel="rbf",C=1.0,gamma="scale"))])

#hard voting
model=VotingClassifier(estimators=[("LR",lr),("RF",rf),("SVM",svm)],voting="hard")

#train
model.fit(x_train,y_train)
print("training successfully")

#test
y_pred=model.predict(x_test)

#ket qua
print("KET QUA VOTE")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1-score:",f1_score(y_test,y_pred))
#ma trận, xem mô hình dự đoán đúng và sai bao nhiêu bệnh nhân, đúng bao nhiêu ng không mắc bệnh, nhầm bao nhiêu người khỏe mạnh thành mắc bệnh, bỏ sót bao nhiêu người mắc bệnh, đúng bao nhiêu người mắc bệnh
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
#classification report báo cáo đánh gí chi tiết cho từng lớp 0 và 1
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
