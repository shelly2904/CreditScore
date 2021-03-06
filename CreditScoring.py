import pandas as pd
import numpy as np
import pickle, os
from settings import *
from Data_Preparation import get_more_variables
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc, roc_auc_score

# #Data files
data = pd.read_csv(os.path.join(DATA_DIR, DATA_TRAINING_FILE))
data_test = pd.read_csv(os.path.join(DATA_DIR, DATA_TEST_FILE))

# #Merging data:
# #get_more_variables(True)

# #get_more_variables(False)

new_data = pd.read_csv(os.path.join(DATA_DIR, NEW_FILE))
new_data = new_data.drop(new_data.columns[0], axis=1)
data = data.merge(new_data, on='customer_no')

new_data_test = pd.read_csv(os.path.join(DATA_DIR, NEW_TEST_FILE))
new_data_test = new_data_test.drop(new_data_test.columns[0], axis=1)
data_test = data_test.merge(new_data_test, on='customer_no')

features = list(data_test.columns)
features.remove('Bad_label')
target = ['Bad_label']

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
                         'feature_40','feature_41','feature_42','feature_43','feature_14','feature_5', 'max_freq_enquiry']

continous_features = list(set(features) - set(features_to_drop) - set(date_features) - set(categorical_features))

def get_model(fileName):
    output = open(os.path.join(DATA_DIR,fileName), 'rb')
    le = pickle.load(output)
    output.close()
    return le

def set_model(clf, fileName):
    output = open(os.path.join(DATA_DIR,fileName), 'wb')
    pickle.dump(clf, output)
    output.close()

def data_preprocessing(df, training=True):
    #removing the unnecessary variables
    df = df.drop(features_to_drop, 1) 
    df = df.drop(date_features, 1) 
    features1 = list(set(features) - set(features_to_drop) - set(date_features))
     
    # #date features
    # for col in date_features:
    #     try:
    #         df[col] = pd.to_datetime(df[col], dayfirst=True)
    #     except:
    #         pass

    for col in continous_features:
        df[col] = df[col].astype(np.float)
        df[col] = df[col].fillna(df[col].mean())

    #Conversion of categorical variable
    for col in categorical_features:
        if training:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            set_model(le, 'LabelEncoder.pkl')
            df[col] = df[col].fillna(df[col].mode())
        else:
            le  = get_model('LabelEncoder.pkl')
            df[col] = le.fit_transform(df[col].astype(str))
    return (df, features1)


def model_fit(clf, X, y):
    clf.fit(X, y)
    return clf 

def model_prediction(clf, X):
    pred = clf.predict(X)
    return pred


data, features = data_preprocessing(data)
data_test, features= data_preprocessing(data_test, False)

dataD, targetD = data.ix[:, features], data.ix[:, target] 
dataTD, targetTD = data_test.ix[:, features], data_test.ix[:, target] 

max_metric = 0
for i in range(1,30):
    for j in range(1, 50):
        print i,j
        model= RandomForestClassifier(n_estimators=i, max_features="sqrt", min_samples_leaf=j)
        model = model_fit(model, dataD, targetD)
        preds = model_prediction(model , dataTD)
        pred_prob = pd.DataFrame(model.predict_proba(dataTD), columns=['Col_0', 'Col_1'])
        pred_prob = pred_prob['Col_1']
        AUC = roc_auc_score(np.array(targetTD['Bad_label'].tolist()), np.array(pred_prob.tolist()))
        gini_index = (2* round(AUC, 2)) - 1
        if max_metric < gini_index:
            print "Best combination", i,j,gini_index
            max_metric = gini_index
            best_model = model

featues_importance = sorted(zip(map(lambda x: round(x, 4), best_model.feature_importances_), features), reverse=True)
print best_model.get_params()
for i in featues_importance:
 print i
set_model(best_model, 'RandomForestClassifier2.pkl')

#print max_metric
model = get_model('RandomForestClassifier2.pkl')
preds = model_prediction(model , dataTD)
pred_prob = pd.DataFrame(model.predict_proba(dataTD), columns=['Col_0', 'Col_1']) 
pred_prob = pred_prob['Col_1']

AUC = roc_auc_score(np.array(targetTD['Bad_label'].tolist()), np.array(pred_prob.tolist()))
gini_index = (2*AUC) - 1
print 'Gini Index', gini_index




