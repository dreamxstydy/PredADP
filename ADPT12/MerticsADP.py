import sys
sys.path.append('../')
Randon_seed = 10
import os
import time
Randon_seed = 10
if not os.path.exists("results_classification"):
    os.mkdir("results_classification")
time_now = int(round(time.time() * 1000))
time_now = time.strftime("%Y-%m-%d_%H-%M", time.localtime(time_now / 1000))
cls_dir1_train = "results_classification/ADP12/Train/{}".format(time_now)
cls_dir1_test = "results_classification/ADP12/Test/{}".format(time_now)
os.makedirs(cls_dir1_train)
os.makedirs(cls_dir1_test)

from sklearn.model_selection import  StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Randon_seed)
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import roc_curve
def score_threshold(y_test,y_pred):
    fpr, tpr, thresholds =roc_curve(y_test, y_pred)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    return threshold
def scores(y_test,y_pred,th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SEN = tp * 1. / (tp + fn)
    SPE = tn * 1. / (tn + fp)
    MCC = matthews_corrcoef(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)

    return [SEN, SPE, MCC, Acc, AUC]



def train_result(file,file_name,model_name):
    SEN = []
    SPE = []
    MCC = []
    ACC = []
    AUC = []

    for i in range(len(file)):
        SEN.append(file[i][0])
        SPE.append(file[i][1])
        MCC.append(file[i][2])
        ACC.append(file[i][3])
        AUC.append(file[i][4])

    with open(os.path.join(cls_dir1_train, file_name), 'w') as f:
        f.write(
            'Datasets' + '    ' + 'SEN' + '     ' + 'SPE' + '    ' + 'F2' + '  ' + 'MCC' + '   ' + 'ACC' + '   ' + 'AUC' + '  ' + 'AUPR' + '\n')
        f.write(str(model_name)  + '  ' + str(format(np.mean(SEN), '.3f')) + ' ' + str(
            format(np.mean(SPE), '.3f')) + ' ' + str(
            format(np.mean(MCC), '.3f')) + ' ' + str(format(np.mean(ACC), '.3f')) + ' ' + str(
            format(np.mean(AUC), '.3f')) + '\n')


def write_test(model_name,file,result_name):
    with open(os.path.join(cls_dir1_test,result_name),'w') as f:
        f.write('model' + '    ' + 'SEN' + '     ' + 'SPE' + '      '+ 'MCC' + '   ' + 'ACC' + '   ' + 'AUC'  +'\n')
        f.write(str(model_name)  + '  ' + str(format(file[0], '.3f')) + ' ' + str(
                format(file[1], '.3f'))  + ' ' + str(format(file[2], '.3f')) + ' ' + str(format(file[3], '.3f')) +' ' + str(format(file[4], '.3f'))  +'\n')