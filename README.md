# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import packages.
2. Read the csv file for datas.
3. Check for duplicate data in the given data.
4. Using sklearn transform every column.
5. assign a column to x.
6. Train the model using test datas.
7. Using logistic regression predict the datas .
8. End. 

## Program:
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()
x=data1.iloc[:,:-1]
print(x)
y=data1["status"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train) 
y_pred=lr.predict(x_test)
print(y)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report 
classification=classification_report(y_test,y_pred)
print(classification)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vijay Ganesh N
RegisterNumber:  212221040177
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](https://github.com/vijayganeshn96/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/exp%203.1.png)
![the Logistic Regression Model to Predict the Placement Status of Student](https://github.com/vijayganeshn96/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/exp%203.2.png)
![the Logistic Regression Model to Predict the Placement Status of Student](https://github.com/vijayganeshn96/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/exp3.3.png)
![the Logistic Regression Model to Predict the Placement Status of Student](https://github.com/vijayganeshn96/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/exp%203.4.png)
![the Logistic Regression Model to Predict the Placement Status of Student](https://github.com/vijayganeshn96/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/blob/main/exp%203.5.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
