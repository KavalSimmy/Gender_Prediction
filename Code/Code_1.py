# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 17:52:06 2021

@author: singh
"""

import pandas as pd
import numpy as np
import matplotlib as pyplot
#%matplotlib inline
import seaborn as sns
data = pd.read_csv('D:\Stevens\SEM_1\MIS 637 - Data Analytics and Machine learning\Final_Project\Gender_Prediction_ML\Data\BlackFriday.csv')
data.info()
data.head()
sns.countplot(x='Gender', data=data)
data.loc[:,'Gender'].value_counts()
sns.countplot(data['Age'], hue=data['Gender'])
data['User_ID'].nunique()
data['Product_ID'].nunique()
def unique(column):
    x=np.array(column)
    print(np.unique(x))
print("Unique ID numbers of customers occupations:")
unique(data['Occupation'])
occupations_id = list(range(0,21))
spent_money = []
for oid in occupations_id:
    spent_money.append(data[data['Occupation'] == oid]['Purchase'].sum())
spent_money
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

objects = ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20')
y_pos = np.arange(len(objects))

plt.bar(y_pos, spent_money, align= 'center', alpha=0.5 )
plt.xticks(y_pos, objects)
plt.ylabel('Money spent')
plt.title('Ocupation ID')

plt.show()
data = data.drop(['Product_Category_3'], axis=1)
data['Product_Category_2'].fillna((data['Product_Category_2'].mean()), inplace=True)
data.info()
data_knn = data[['Occupation', 'Gender', 'Purchase']]
data_knn.head()
data_knn.info()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data_knn.loc[:, data_knn.columns != 'Gender'], data_knn.loc[:,'Gender']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Accuracy:", knn.score(x_test, y_test))