# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

##Algorithm
1.import pandas module.
2.Read the required csv file using pandas.
3.Import LabEncoder module.
4.From sklearn import logistic regression.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.print the required values.
8.End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SYAM TEJ.P
RegisterNumber:  212221240056
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
print(data.head())
data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)
print(data1.head())
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])
print(data1)
x = data1.iloc[:,:-1]
print(x)
y = data1["status"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```

## Output:
1.![4 1](https://user-images.githubusercontent.com/93427224/162664192-2ffc46e8-6541-4dde-beb7-b652337a19e6.jpg)
2.![4 2](https://user-images.githubusercontent.com/93427224/162664205-42684eb1-3478-4560-abe5-1840426511e6.jpg)
3.![4 3](https://user-images.githubusercontent.com/93427224/162664220-1b86d14f-34e3-474f-b2ad-53cb0fb79eee.jpg)
4.![4 4](https://user-images.githubusercontent.com/93427224/162664237-fcf3d0e4-40a6-4c70-939d-96f50f1c0468.jpg)
5.![4 5](https://user-images.githubusercontent.com/93427224/162664503-1ed36d47-8e21-4f64-9f57-dd7cd30ac1b2.jpg)
6.![4 6](https://user-images.githubusercontent.com/93427224/162664515-0ed26301-ab56-4a02-82f5-982d2593e23a.jpg)
7.![4 7](https://user-images.githubusercontent.com/93427224/162664533-dba4df99-d98b-4d4a-81bc-424bf07f9c35.jpg)
8.![4 8](https://user-images.githubusercontent.com/93427224/162664545-5d76b43c-8d5d-416b-bb0f-4acc3581c414.jpg)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
