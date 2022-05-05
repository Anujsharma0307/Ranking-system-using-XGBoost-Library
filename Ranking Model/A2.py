#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import statements
import xgboost as xgb


# In[2]:


import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


#Data preparations
#importing the data file in the form of a dataframe
df=pd.read_csv("/Users/ANUJ/Desktop/practical data science/A2_data/train.tsv",sep="\t")


# In[6]:


df.head()


# In[7]:


#checking the statistics of the dataframe
df.describe()


# In[8]:


#Rename the first column to remove the '#' sign
df = df.rename(columns={'#QueryID': 'QueryID'})


# In[9]:


df.head()


# In[10]:


#Checking for null values
df.isnull().sum()


# In[11]:


#checking if there are  any different values
df['QueryID'].value_counts()


# In[12]:


df['Docid'].value_counts()


# In[13]:


df['Label'].value_counts()


# In[14]:


#The data seems to be clean and no major errors seem to be present.and is ready for exploration.


# In[15]:


a=df['QueryID']
a.describe()


# In[16]:


b=df['Docid']
b.describe()


# In[17]:


#create a separate dataframe for just 'Queryid','Docid' and 'Label' 
X_drop=df.drop(['QueryID','Docid','Label'], axis=1)
Y_parameter=df['Label']


# In[18]:


Y_parameter.describe()


# In[19]:


#separarting testing dataset from training dataset
X_train, X_test, Y_train,Y_test =train_test_split(X_drop,Y_parameter,test_size=0.35, random_state=5)


# In[20]:


len(X_train)


# In[21]:


len(X_test)


# In[22]:


len(Y_train)


# In[23]:


len(Y_train)


# In[24]:


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[25]:


#Training our model
df_train = xgb.DMatrix(X_train, Y_train)
df_train2 =xgb.DMatrix(X_test, Y_test)


# In[26]:


#Setting up parameters
module_parameters={'objective':'rank:ndcg',
'learning_rate' : 0.1,
'min_child_weight':0.1,
'max_depth':5,'eval_metric':'ndcg@100'}


# In[27]:


print(module_parameters)


# In[28]:


#Train the dataset 
training_set = [(df_train,'train'),(df_train2,'val')]
total_Predictions = 100
XGtrainingmodel = xgb.train(module_parameters, df_train, total_Predictions, training_set)


# In[29]:


#Checking the prediction with predict() function
Ytraining_prediction = XGtrainingmodel.predict(xgb.DMatrix(X_train))
Yvalidation_prediction = XGtrainingmodel.predict(xgb.DMatrix(X_test))


# In[30]:


print(Ytraining_prediction)


# In[31]:


print(Yvalidation_prediction)


# In[32]:


#Importing the testing dataset
testing_Data = pd.read_csv("/Users/ANUJ/Desktop/practical data science/A2_data/test.tsv", sep='\t')
testing_Data.head()


# In[33]:


testing_Data = testing_Data.rename(columns={'#QueryID': 'QueryID'})


# In[34]:


df3=testing_Data.drop(['QueryID','Docid'], axis=1)
df3.head()


# In[35]:


df_test = xgb.DMatrix(df3)
prediction = XGtrainingmodel.predict(df_test)
print(prediction)


# In[36]:


#Formating the runs in the format
Query_test = testing_Data["QueryID"]
Docid_test = testing_Data["Docid"]
Df_Testing=testing_Data.groupby('QueryID')['QueryID'].count().to_numpy()
Df_prediction = pd.DataFrame({'QueryID':Query_test,'Docid':Docid_test,'Score':prediction})
print(Df_prediction)


# In[37]:


#print out the runs in the form of a tsv file
Df_prediction.to_csv('A2run.txt', sep='\t', index=False)


# In[ ]:




