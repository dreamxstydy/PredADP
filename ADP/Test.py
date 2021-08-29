import pandas as pd
import joblib
import numpy as np
from .Mertics import scores,write_test

def test(file):
        TEAAC  = file[1]
        y = file[0]
        y = np.array([1 if i < 102 else 0 for i in range(205)])
        y = pd.DataFrame(y,columns=['class'])
        model = joblib.load('./Models/Train/AAC.model')
        TEAAC = model.predict_proba(TEAAC)[:,1]
        TEAAC = pd.DataFrame(TEAAC,columns=['AAC'])
        TEBPNC = file[2]
        model = joblib.load('./Models/Train/BPNC.model')
        TEBPNC = model.predict_proba(TEBPNC)[:,1]
        TEBPNC = pd.DataFrame(TEBPNC,columns=['BPNC'])
        TECTD = file[3]
        model = joblib.load('./Models/Train/CTD.model')
        TECTD = model.predict_proba(TECTD)[:,1]
        TECTD = pd.DataFrame(TECTD,columns=['CTD'])
        TEDPC = file[4]
        model = joblib.load('./Models/Train/DPC.model')
        TEDPC = model.predict_proba(TEDPC)[:,1]
        TEDPC = pd.DataFrame(TEDPC,columns=['DPC'])
        Test = pd.concat([y,TEAAC,TEBPNC,TECTD,TEDPC],axis=1,join='inner')
        return Test


def test_best(file):
        X = file.iloc[:, 1:]
        y = file.iloc[:, 0]
        model_name = ["LR","ERT","RF","KNN"]
        result_name = ["LR.txt","ERT.txt","RF.txt","KNN.txt"]
        for i in range(len(model_name)):
                model = joblib.load(f'./Models/Train/bestTrain/{model_name[i]}.model')
                test_pred = model.predict_proba(X)[:, 1]
                score = scores(y, test_pred)
                write_test(model_name[i],score,result_name[i])
