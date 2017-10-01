
# coding: utf-8

# In[20]:

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn import linear_model
from settings import *
from Data_Preparation import get_more_variables


# In[2]:

#Data files
data = pd.read_csv(os.path.join(DATA_DIR, DATA_TRAINING_FILE))
data_test = pd.read_csv(os.path.join(DATA_DIR, DATA_TEST_FILE))


# In[3]:

#Merging data:
get_more_variables(True)


# In[4]:

get_more_variables(False)


# In[5]:

new_data = pd.read_csv(os.path.join(DATA_DIR, NEW_FILE))
data = data.merge(new_data, on='customer_no')

new_data_test = pd.read_csv(os.path.join(DATA_DIR, NEW_TEST_FILE))
data_test = data_test.merge(new_data_test, on='customer_no')


# In[6]:

features = list(data.columns)
features.remove('Bad_label')
target = ['Bad_label']


# In[7]:

features_to_drop = ['customer_no', 'feature_20', 'feature_45', 'feature_18','feature_15','feature_17',
                     'feature_47', 'feature_22', 'feature_77', 'feature_24','feature_16','feature_10','feature_61']

date_features = ['dt_opened','feature_53','feature_63','entry_time',
                  'feature_70','feature_75','feature_54','feature_21','feature_2']

categorical_features = ['feature_67','feature_68','feature_72','feature_73','feature_74','feature_28',
                         'feature_76','feature_78','feature_11','feature_79','Bad_label','feature_46',
                         'feature_48','feature_50','feature_51','feature_4','feature_1','feature_8',
                         'feature_9','feature_12','feature_19','feature_23','feature_25','feature_30',
                         'feature_31','feature_32','feature_33','feature_26','feature_13','feature_55',
                         'feature_57','feature_58','feature_59','feature_60','feature_62','feature_36',
                         'feature_6','feature_34','feature_37','feature_27','feature_38','feature_39',
                         'feature_40','feature_41','feature_42','feature_43','feature_14','feature_5']

continous_features = list(set(features) - set(features_to_drop) - set(date_features) - set(categorical_features))


# In[8]:

#removing the unnecessary varibles
data = data.drop(features_to_drop, 1) 
data_test = data_test.drop(features_to_drop, 1)
features = list(set(features) - set(features_to_drop))

data = data.drop(date_features, 1)  
data_test = data_test.drop(date_features, 1)  
features = list(set(features) - set(date_features))


# In[10]:

# #date features
# for col in date_features:
#     try:
#         data[col] = pd.to_datetime(data[col], dayfirst=True)
#     except:
#         pass

for col in continous_features:
    data[col] = data[col].astype(np.float)
    data_test[col] = data_test[col].astype(np.float)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
mean_imputer = mean_imputer.fit(data[continous_features])
data[continous_features] = mean_imputer.fit_transform(data[continous_features].values)
data_test[continous_features] = mean_imputer.fit_transform(data_test[continous_features].values)


# In[11]:

#Conversion of categorical variable
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    data_test[col] = le.fit_transform(data_test[col].astype(str))
    #data[col] = le.transform(data[col])
    data[col] = data[col].fillna(data[col].mode())
    data_test[col] = data_test[col].fillna(data_test[col].mode())
    


# In[12]:

dataD = data.ix[:, features]
targetD = data.ix[:, target] 

print dataD.describe()
print targetD.describe()


# In[18]:

# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(data, targetD)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


# In[19]:

dataD = data.ix[:,fit.support_]
targetD = data.ix[:, target] 

dataTD = data_test.ix[:,fit.support_]
targetTD = data_test.ix[:, target] 

#Initialize logistic regression model
log_model = linear_model.LogisticRegression()

# Train the model
log_model.fit(X = dataD , y = targetD)

# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)

# Make predictions
preds = log_model.predict(X= dataTD)

# Generate table of predictions vs actual
print pd.crosstab(preds,targetTD)




# In[22]:

output = open(os.path.join(DATA_DIR, 'LogisticRegressor.pkl'), 'wb')
pickle.dump(log_model, output)
output.close()


# In[ ]:



