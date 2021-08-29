import pandas as pd
from  sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.linear_model.logistic import LogisticRegression as LR
import numpy as np
def scores(y_test,y_pred,th=0.5):
    y_predlabel=[(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp=confusion_matrix(y_test,y_predlabel).flatten()
    SEN=tp*1./(tp+fn)
    SPE=tn*1./(tn+fp)
    MCC=matthews_corrcoef(y_test,y_predlabel)
    Acc=accuracy_score(y_test, y_predlabel)
    AUC=roc_auc_score(y_test, y_pred)
    return [SEN,SPE,MCC,Acc,AUC]

def othertest(filebest):
    for i in range(filebest.shape[1] -1):
        data1 = filebest.iloc[:,:i+1]
        data2 = filebest.iloc[:,i+2:]
        data3 = pd.concat([data1,data2],axis=1,join='inner')
        X = data3.iloc[:,1:].values
        y = data3.iloc[:,0].values
        skf = StratifiedKFold(n_splits=5)
        merticsl = []
        for train_index, test_index in skf.split(X, y):
            train_X,test_X = X[train_index], X[test_index]
            train_Y, test_Y = y[train_index], y[test_index]
            clf = LR(random_state=10)
            clf = clf.fit(train_X, train_Y)
            y_pred = clf.predict_proba(test_X)[:, 1]
            scorel = scores(test_Y, y_pred)
            merticsl.append(scorel)
            SEN = []
            SPE = []
            MCC = []
            ACC = []
            AUC = []
            for i in range(len(merticsl)):
                SEN.append(merticsl[i][0])
                SPE.append(merticsl[i][1])
                MCC.append(merticsl[i][2])
                ACC.append(merticsl[i][3])
                AUC.append(merticsl[i][4])
        print("SEN:%s,SPE:%s,MCC:%s,ACC:%s,AUC:%s" % ( np.mean(SEN), np.mean(SPE), np.mean(MCC), np.mean(ACC), np.mean(AUC)))