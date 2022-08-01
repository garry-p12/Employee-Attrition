#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
from time import strftime
import tensorflow as tf
import keras
import requests
import json
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.svm import SVC
#from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten, Conv2D, MaxPooling2D
from keras.layers import *
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
get_ipython().run_line_magic('reload_ext', 'tensorboard')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('employees (1) (1) (1) (2).csv')
data


# In[3]:


data.isnull().sum()


# In[4]:


data['last_evaluation'].fillna('0',inplace=True)


# In[5]:


data.isnull().sum()


# In[6]:


data['department'].fillna('support',inplace=True)


# In[7]:


data['satisfaction'].fillna('0.0',inplace=True)


# In[8]:


data['tenure'].fillna('0',inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


le = LabelEncoder()


# In[11]:


label = le.fit_transform(data['EmployeeName'])
label


# In[12]:


label = le.fit_transform(data['Agency'])
label


# In[13]:


label = le.fit_transform(data['fname'])
label


# In[14]:


label = le.fit_transform(data['lname'])
label


# In[15]:


data.drop("EmployeeName", axis=1, inplace=True)


# In[16]:


data.drop("Agency", axis=1, inplace=True)


# In[17]:


data.drop("fname", axis=1, inplace=True)


# In[18]:


data.drop("lname", axis=1, inplace=True)


# In[19]:


data["EmployeeName"] = label


# In[20]:


data["Agency"] = label


# In[21]:


data["fname"] = label


# In[22]:


data["lname"] = label


# In[23]:


data.head()


# In[24]:


data['status']=data['status'].map({'Employed':1,'Left':0})
data.head()


# In[25]:


data['salary']=data['salary'].map({'low':0,'medium':1,'high':2})
data.head()


# In[26]:


data['department']=data['department'].map({'product':0,'sales':1,'support':2,'temp':3,'IT':4,'admin':5,'engineering':6,'finance':7,'information_technology':8,'management':9,'marketing':10,'procurement':11})


# In[27]:


data.groupby('department')['status'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[28]:


data.groupby('n_projects')['status'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[29]:


data.groupby('salary')['status'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[30]:


data.groupby('tenure')['status'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[31]:


data.groupby('status')['status'].count().plot(kind='pie',autopct='%1.1f%%',shadow=True,figsize=(7,7))


# In[ ]:





# In[ ]:





# In[ ]:





# In[32]:


data.corr().head()


# In[33]:


plt.figure(figsize=(16,10))
sns.heatmap(data.corr(), annot=True, annot_kws={"size": 14})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[34]:


plt.figure(figsize=(7,7))
sns.kdeplot(data['salary'],shade=True)
plt.show()


# In[35]:


plt.figure(figsize=(7,7))
sns.kdeplot(data['avg_monthly_hrs'],shade=True)
plt.show()


# In[36]:


sns.pairplot(data)


# In[37]:


data = data.drop(['Agency'],axis=1)


# In[38]:


data = data.drop(['EmployeeName'],axis=1)


# In[39]:


data = data.drop(['fname'],axis=1)


# In[40]:


data=data.drop(['lname'],axis=1)


# In[41]:


data = data.drop(['avg_monthly_hrs'],axis=1)


# In[42]:


data = data.drop(['last_evaluation'],axis=1)


# In[43]:


data.head()


# In[44]:


target = np.array(data.drop(['status'],1)).astype('float32')
features = np.array(data['status']).astype('float32')


# In[45]:


target[0]


# In[46]:


x_train , x_test , y_train , y_test = train_test_split(target,features,stratify=features,test_size=0.25,random_state=42)

len(x_train)/len(features)


# In[47]:


print(f'Shape of x_train is {x_train.shape}')
print(f'Shape of x_test is {x_test.shape}')
print(f'Shape of y_train is {y_train.shape}')
print(f'Shape of y_test is {y_test.shape}')


# In[48]:


rc = RandomForestClassifier()
rc.fit(x_train,y_train)
ypred = rc.predict(x_test)
print('Random Forest Classifier',":",accuracy_score(y_test,ypred)*100)


# In[49]:


lr = LogisticRegression()
lr.fit(x_train,y_train)
ypred1 = lr.predict(x_test)
print('Logisitic Regression',":",accuracy_score(y_test,ypred1)*100)


# In[50]:


dc = DecisionTreeClassifier()
dc.fit(x_train,y_train)
ypred3 = dc.predict(x_test)
print('Decision tree classifier',":",accuracy_score(y_test,ypred3)*100)


# In[51]:


svr = SVC()
svr.fit(x_train,y_train)
ypred2 = svr.predict(x_test)
print('Support Vector Classifier',":",accuracy_score(y_test,ypred2)*100)


# In[52]:


model_1 = Sequential([
    Dense(units=64,input_dim=8,activation='relu',name='m1_hidden1'),
    Dense(units=16,activation='relu',name='m1_hidden2'),
    Dense(8,activation='relu',name='m1_hidden3'),
    Dense(2,activation='softmax',name='output')
    
])
model_1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[53]:


model_1.summary()


# In[54]:


LOG_DIR = 'tensorboard_attrition_logs/'
def get_tensorboard(model_name):
    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths =os.path.join(LOG_DIR,folder_name)
    try:
        os.makedirs(dir_paths)
    except OSError as err:
        print(err.strerror)
    else:
        print('Successfully created directory')
    return TensorBoard(log_dir = dir_paths)    


# In[55]:


samples_per_batch = 32


# In[56]:


nr_epochs = 100
history=model_1.fit(x_train , y_train,batch_size=samples_per_batch,epochs=nr_epochs,callbacks=[get_tensorboard('Model_1')],verbose=0,validation_data=(x_test,y_test))


# In[57]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[58]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[59]:


from sklearn.metrics import classification_report, accuracy_score

categorical_pred = np.argmax(model_1.predict(x_test), axis=1)

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))


# In[60]:


Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print(Y_train_binary[:20])


# In[61]:


def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    #adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()


# In[62]:


binary_model.summary()


# In[63]:


history=binary_model.fit(x_train, Y_train_binary, validation_data=(x_test, Y_test_binary), epochs=50,verbose=0, batch_size=10)


# In[64]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[65]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[66]:


binary_pred = np.round(binary_model.predict(x_test)).astype(int)

print('Results for Binary Model')
print(round(accuracy_score(Y_test_binary, binary_pred)*100))


# In[67]:


cdf = data[['age','rating','n_projects','salary','salary.1','tenure','satisfaction','status']]


# In[68]:


x = cdf.iloc[:,:3]
y = cdf.iloc[:,-1]


# In[78]:


import pickle
pickle.dump(lr,open('employee_attrition_1.pkl','wb'))


# In[79]:


model = pickle.load(open('employee_attrition_1.pkl','rb'))


# In[80]:


print(model.predict([[6,4,0,0.829896,5,43,1.4,567695]]))


# In[81]:


print(model.predict([[5,2,0,0.469896,2,23,3.4,367695]]))


# In[83]:


print(model.predict([[1,1,0,0.629896,2,33,4.4,347695]]))


# In[84]:


print(model.predict([[1,1,0,0.129896,2,33,1.4,147695]]))


# In[ ]:




