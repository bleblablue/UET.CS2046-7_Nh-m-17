import pandas
import numpy

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#đọc dữ liệu
df=pandas.read_csv("pima-indians-diabetes.csv")

#đặt tên cho các cột
df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
#xử lý giá trị 0
zero=[ "Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

df[zero]=df[zero].replace(0,numpy.nan)

#tách đặc trưng và nhãn
x=df.drop("Outcome",axis=1)
y=df["Outcome"]

#chia train test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

#pine line
model=Pipeline([
    ("imputer", KNNImputer(n_neighbors=10)), #ước lượng thay thế giá trị thiếu
    ("scaler",StandardScaler()), #chuấn hóa đặc trưng
    ("smote",SMOTE(random_state=42,k_neighbors=5)),
    ("classifier",RandomForestClassifier(n_estimators=100,random_state=42))
])

#train mô hình
model.fit(x_train,y_train)
#test
y_pred=model.predict(x_test)

#kết quả
print("SMOTE + RANDOM FOREST")
print("Accuracy:",accuracy_score(y_test, y_pred))
print("Precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
print("F1-score:",f1_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))