# các giá trị glucose, blood pressure, skin thichness, insulin và BMI 
# là 0 được coi là bất thường và cần chuyển về N/A

import pandas #đọc file cvs bằng pandas
import numpy #tính toán khoa học
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
#đọc dữ liệu 
data = pandas.read_csv(r"C:\Users\Admin\OneDrive\Documents\baiAI\UET.CS2046-7_Nh-m-17\pima-indians-diabetes.csv", header=None) # đường dẫn tuyệt đối của file trong máy
#đặt tên cho các cột
data.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# Thay thế 0 bằng NaN
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns] = data[columns].replace(0, numpy.nan)

#sử dụng 8 thuộc tính đầu vào => không sử dụng cột outcome
x=data.drop(columns=["Outcome"])

scaler=StandardScaler()
X_scaled=scaler.fit_transform(x) #đưa các thuộc tính về thang đo có trung bình gần với 0 và độ lệch chuẩn gần 1
#=> các thuộc tính có giá trị lớn không thể chi phối khoảng cách

#KNN
imputer=KNNImputer(n_neighbors=10, weights="distance") #weights : các th gần nhau có ảnh hướng lớn hơn
X_imputed=imputer.fit_transform(X_scaled) #tìm các th gần nhau và thực hiện thay thế giá trị

#chuyển dữ liệu lại về thang đo ban đầu
X_imputed=scaler.inverse_transform(X_imputed)
#cập nhật dữ liệu
data[x.columns]=X_imputed

