
import pandas as pd
import  numpy as np
import joblib
from  pathlib import  Path
from sklearn.model_selection import  StratifiedKFold
from xgboost.sklearn import XGBClassifier as XGBoost

import os
import sys

sys.path.append('../')


Randon_seed = 10
njobs = 8
Path('./Models/TrainADP/').mkdir(exist_ok = True,parents = True)

def base_clf(clf,X_train,y_train,model_name,n_folds=5):
    ntrain = X_train.shape[0]
    nclass = len(np.unique(y_train))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Randon_seed)
    base_train = np.zeros((ntrain,nclass))
    for train_index, test_index in kf.split(X_train,y_train):
        kf_X_train,kf_y_train = X_train[train_index],y_train[train_index]
        kf_X_test = X_train[test_index]
        clf.fit(kf_X_train, kf_y_train)
        base_train[test_index] = clf.predict_proba(kf_X_test)
    clf.fit(X_train,y_train)
    joblib.dump(clf,f'./Models/TrainADP/{model_name}.model')
    return base_train[:,-1]





def StackADPfeature(file):
    final_Features = []
    feature_Name = ['AAC', 'BPNC', 'CTD', 'DPC']
    y = file[0]
    y = np.array([1 if i < 248 else 0 for i in range(407)])
    for j in range(1, len(file)):
        X = file[j]
        final_Features.append(base_clf(XGBoost(), X, np.array(y), feature_Name[j - 1]))
    Features = pd.DataFrame(np.array(final_Features).T, columns=feature_Name)
    y = pd.DataFrame(y,columns=['class'])
    TRFeatures = pd.concat([y,Features],axis=1,join='inner')
    file_path = ('ADPT12/Train/ADPT12Stack.csv')
    TRFeatures.to_csv(file_path,index=False)
    return file_path


