#chia dữ liệu 80% training và 20% test
import pandas
from sklearn.model_selection import train_test_split

#đọc dữ liệu đã được xử lý
data = pandas.read_csv("UET.CS2046-7_Nh-m-17/du_lieu_tien_xu_ly.csv")
#chia thuộc tính và nhãn
x=data.drop(columns=["Outcome"]) #thuộc tính
y=data["Outcome"] #nhãn bệnh
#chia dữ liệu
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y) #random: mỗi lần chạy đều chia dữ liệu giống nhau, stratify=y tỷ lệ mắc bệnh và k tương đương giữa 2 tập sau khi chia
#ghép thuộc tính và nhãn
train_data=x_train.copy()
train_data["Outcome"]=y_train

test_data=x_test.copy()
test_data["Outcome"]=y_test

#lưu thành 2 file riêng
train_data.to_csv("UET.CS2046-7_Nh-m-17/train.csv",index=False)
test_data.to_csv("UET.CS2046-7_Nh-m-17/test.csv",index=False)

print("bmw")