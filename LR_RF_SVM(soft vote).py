#soft vote
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

# đọc dữ liệu
train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")

# tách thuộc tính và nhãn
x_train = train_data.drop(columns=["Outcome"])
y_train = train_data["Outcome"]
x_test = test_data.drop(columns=["Outcome"])
y_test = test_data["Outcome"]
# lr
lr = Pipeline([("scaler", StandardScaler()),("model", LogisticRegression(max_iter=1000))])

# rf
rf = RandomForestClassifier(n_estimators=100,random_state=42)

# svm
svm = Pipeline([("scaler", StandardScaler()),("model", SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,random_state=42))])

# vote
model = VotingClassifier(estimators=[("LR", lr),("RF", rf),("SVM", svm)],voting="soft")

# train
model.fit(x_train, y_train)

print("training successfully")

# test
y_pred = model.predict(x_test)

# đánh giá
print("KET QUA")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

# ma trận
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# báo cáo
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
