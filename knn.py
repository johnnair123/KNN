import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report

ads_df=pd.read_csv('https://raw.githubusercontent.com/safal/DS-
ML/refs/heads/main/Social_Network_Ads.csv')

print(ads_df.head())
print(ads_df.isnull().sum())
features=ads_df.drop(['User ID','Purchased'],axis=1)
target=ads_df['Purchased']
print(features.head())
features_convert= pd.get_dummies(features)
print(features_convert.head())
x_train,x_test,y_train,y_test=train_test_split(features_convert,target,test_size=0.3,ran
dom_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
ob=KNeighborsClassifier(n_neighbors=5)
KNN_model=ob.fit(x_train,y_train)
y_train_predict=KNN_model.predict(x_train)
y_test_predict=KNN_model.predict(x_test)
print("Accuracy of train data:",accuracy_score(y_train,y_train_predict))
print("Accuracy of test data:",accuracy_score(y_test,y_test_predict))
print(classification_report(y_test,y_test_predict))