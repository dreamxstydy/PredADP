import pandas as pd
import joblib
import numpy as np
import sys
from .MerticsADP import scores,write_test
sys.path.append('../')
def testADPT12(file):
        y = file[0]
        y = np.array([1 if i < 62 else 0 for i in range(103)])
        y = pd.DataFrame(y,columns=['class'])
        model = joblib.load('./Models/TrainADP/AAC.model')
        TEAAC = file[1]
        TEAAC = model.predict_proba(TEAAC)[:,1]
        TEAAC = pd.DataFrame(TEAAC,columns=['AAC'])
        TEBPNC = file[2]
        model = joblib.load('./Models/TrainADP/BPNC.model')
        TEBPNC = model.predict_proba(TEBPNC)[:,1]
        TEBPNC = pd.DataFrame(TEBPNC,columns=['BPNC'])
        TECTD = file[3]
        model = joblib.load('./Models/TrainADP/CTD.model')
        TECTD = model.predict_proba(TECTD)[:,1]
        TECTD = pd.DataFrame(TECTD,columns=['CTD'])
        TEDPC = file[4]
        model = joblib.load('./Models/TrainADP/DPC.model')
        TEDPC = model.predict_proba(TEDPC)[:,1]
        TEDPC = pd.DataFrame(TEDPC,columns=['DPC'])
        Test = pd.concat([y,TEAAC,TEBPNC,TECTD,TEDPC],axis=1,join='inner')
        return Test


def TestADP_best(file):
        X = file.iloc[:, 1:]
        y = file.iloc[:, 0]
        model_name = ["LR","ERT","RF","KNN"]
        result_name = ["LR.txt","ERT.txt","RF.txt","KNN.txt"]
        for i in range(len(model_name)):
                model = joblib.load(f'./Models/TrainADP/bestTrain/{model_name[i]}.model')
                test_pred = model.predict_proba(X)[:, 1]
                score = scores(y, test_pred)
                write_test(model_name[i],score,result_name[i])