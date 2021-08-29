from xgboost.sklearn import XGBClassifier as XGBoost
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression as LR
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
def scores(y_test,y_pred,th=0.5):
    y_predlabel=[(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp=confusion_matrix(y_test,y_predlabel).flatten()
    SEN=tp*1./(tp+fn)
    SPE=tn*1./(tn+fp)
    MCC=matthews_corrcoef(y_test,y_predlabel)
    Acc=accuracy_score(y_test, y_predlabel)
    AUC=roc_auc_score(y_test, y_pred)
    return [SEN,SPE,MCC,Acc,AUC]


def xgbtrainT(file):
     for i in range(1,len(file)):
         X = file[i]
         y = file[0]
         y = np.array([1 if i < 248 else 0 for i in range(407)])
         skf = StratifiedKFold(n_splits=5)
         metrics = []
         for train_index, test_index in skf.split(X, y):
             trainX, testX = X[train_index], X[test_index]
             trainY, testY = y[train_index], y[test_index]
             clf =XGBoost(random_state=10)
             clf = clf.fit(trainX, trainY)
             y_pred = clf.predict_proba(testX)[:, 1]
             score = scores(testY, y_pred)
             metrics.append(score)
             SEN = []
             SPE = []
             MCC = []
             Acc = []
             AUC = []
             for i in range(len(metrics)):
                 SEN.append(metrics[i][0])
                 SPE.append(metrics[i][1])
                 MCC.append(metrics[i][2])
                 Acc.append(metrics[i][3])
                 AUC.append(metrics[i][4])
         print("SEN:%s,SPE:%s,MCC:%s,ACC:%s,AUC:%s" % (np.mean(SEN), np.mean(SPE), np.mean(MCC), np.mean(Acc), np.mean(AUC)))

