from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model.logistic import  LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier as ERT
import joblib
from pathlib import Path
from .MerticsADP import scores,train_result
Path(f'./Models/TrainADP/bestTrain/').mkdir(exist_ok=True, parents=True)
def ADP12L_Model(file):
    file_name = 'LR.txt'
    model_name = 'LR'
    X = file.iloc[:, 1:].values
    y = file.iloc[:, 0].values
    skf = StratifiedKFold(n_splits=5)
    clf = LR(random_state=10)
    result_LR = []
    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = y[train_index], y[test_index]
        clf = clf.fit(train_X,train_Y)
        y_pred = clf.predict_proba(test_X)[:,1]
        scorel = scores(test_Y,y_pred)
        result_LR.append(scorel)
    joblib.dump(clf, f'./Models/TrainADP/bestTrain/{model_name}.model')
    train_result(result_LR,file_name,model_name)




def ADP12ERT_Model(file):
    file_name = 'ERT.txt'
    model_name = 'ERT'
    X = file.iloc[:, 1:].values
    y = file.iloc[:, 0].values
    skf = StratifiedKFold(n_splits=5)
    clf = ERT(random_state=10)
    result_ERT = []
    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = y[train_index], y[test_index]
        clf = clf.fit(train_X, train_Y)
        y_pred = clf.predict_proba(test_X)[:, 1]
        scorel = scores(test_Y, y_pred)
        result_ERT.append(scorel)
    joblib.dump(clf, f'./Models/TrainADP/bestTrain/{model_name}.model')
    train_result(result_ERT,file_name,model_name)


def ADP12R_Model(file):
    file_name = 'RF.txt'
    model_name = 'RF'
    X = file.iloc[:, 1:].values
    y = file.iloc[:, 0].values
    skf = StratifiedKFold(n_splits=5)
    clf = RF(random_state=10)
    result_RF = []
    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = y[train_index], y[test_index]
        clf = clf.fit(train_X, train_Y)
        y_pred = clf.predict_proba(test_X)[:, 1]
        scorel = scores(test_Y, y_pred)
        result_RF.append(scorel)
    joblib.dump(clf, f'./Models/TrainADP/bestTrain/{model_name}.model')
    train_result(result_RF,file_name,model_name)

def ADP12K_Model(file):
    file_name= 'KNN.txt'
    model_name = 'KNN'
    X = file.iloc[:, 1:].values
    y = file.iloc[:, 0].values
    skf = StratifiedKFold(n_splits=5)
    clf = KNN()
    result_KNN = []
    for train_index, test_index in skf.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_Y, test_Y = y[train_index], y[test_index]
        clf = clf.fit(train_X, train_Y)
        y_pred = clf.predict_proba(test_X)[:, 1]
        scorel = scores(test_Y, y_pred)
        result_KNN.append(scorel)
    joblib.dump(clf, f'./Models/TrainADP/bestTrain/{model_name}.model')
    train_result(result_KNN,file_name,model_name)