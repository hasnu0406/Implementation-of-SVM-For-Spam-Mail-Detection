# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect the encoding of the `spam.csv` file and load it using the detected encoding.
2. Check basic data information and identify any null values.
3. Define the features (`X`) and target (`Y`), using `v2` as the feature (message text) and `v1` as the target (spam/ham label).
4. Split the data into training and testing sets (80-20 split).
5. Use `CountVectorizer` to convert the text data in `X` to a matrix of token counts, fitting on the training set and transforming both training and test sets.
6. Initialize and train an SVM classifier on the transformed training data.
7. Predict the target labels for the test set.
8. Calculate and display the model's accuracy.
## Program:
```Python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HASNA MUBARAK AZEEM
RegisterNumber: 212223240052 
*/

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/57baf6fa-ecec-4823-bbf6-00074217f578)

![image](https://github.com/user-attachments/assets/f1f24ed2-ffe5-4fad-a3a8-c68133f75b16)

![image](https://github.com/user-attachments/assets/d3cb49d9-d1d4-4254-a4b5-afc35c4e2e7b)

![image](https://github.com/user-attachments/assets/35de045b-ddd6-4e70-988a-2a766cefbae4)

![image](https://github.com/user-attachments/assets/fc05e94d-a1c2-41ee-b668-1a5005fbecc0)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
