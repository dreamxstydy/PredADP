from  PredADP.ADP.Feature import out
from  PredADP.ADP.XGB import xgbtrain
from  PredADP.ADP.Stack import Stackfeature
from  PredADP.ADP.Train import L_Model, ERT_Model, R_Model, K_Model
from  PredADP.ADP.Test import test, test_best
from  PredADP.ADP.Other import othertest
import pandas as pd
from PredADP.ADPT12.StackADP import StackADPfeature
from PredADP.ADPT12.XGBT12 import xgbtrainT
from PredADP.ADPT12.TrainADP import ADP12L_Model, ADP12ERT_Model, ADP12R_Model, ADP12K_Model
from PredADP.ADPT12.TestADPT12 import TestADP_best, testADPT12

if __name__ == "__main__":
    Train_feature = out('./datasets/Train.txt')
    Train_stack = Stackfeature(Train_feature)
    Teain_fea = pd.read_csv(Train_stack)
    print('ADP/NonADP每个原始特征组基于XGB的性能: ' + '\n')
    FeatureXgb = xgbtrain(Train_feature)
    print("---------------------------------------------------------------------------------")
    print('ADP/NonADP训练集在LR ERT RF KNN分类器依次训练得到的性能：' + '\n')
    LR_result = L_Model(Teain_fea)
    ERT_result = ERT_Model(Teain_fea)
    RF_result = R_Model(Teain_fea)
    KNN_result = K_Model(Teain_fea)
    print('--------------------------------------------------------------------------------')
    print('ADP/NonADP除去自身之外的特征组合在LR分类器训练得到的结果')
    othertest(Teain_fea)
    print('--------------------------------------------------------------------------------')
    print('ADP/NonADP测试集在XGB LR RF KNN分类器依次训练得到的性能：' + '\n')
    Test_Feature = out('./datasets/Test.txt')
    Test_stack = test(Test_Feature)
    test_best(Test_stack)
    print('-----------------------------------------------------------------------------')


    TRADPT12_feature = out('./datasets/TrainADP.txt')
    TrainADPT12_stack = StackADPfeature(TRADPT12_feature)
    TrainADPT12_fea =pd.read_csv(TrainADPT12_stack)

    print('ADPT1/ADPT2每个原始特征组基于XGB的性能: ' + '\n')
    TypeFeatureXgb = xgbtrainT(TRADPT12_feature)
    print('--------------------------------------------------------------------------------')
    print('ADPT1/ADPT2训练集在 LR ERT RF KNN分类器依次训练得到的性能：' + '\n')
    LR_result = ADP12L_Model(TrainADPT12_fea)
    ERT_result = ADP12ERT_Model(TrainADPT12_fea)
    RF_result = ADP12R_Model(TrainADPT12_fea)
    KNN_result = ADP12K_Model(TrainADPT12_fea)
    print('---------------------------------------------------------------------------------')
    print('ADPT1/ADPT2除去自身之外的特征组合在LR分类器训练得到的结果')
    othertest(TrainADPT12_fea)
    print('--------------------------------------------------------------------------------')
    print('ADPT1/ADPT2测试集在LR ERT RF KNN分类器依次训练得到的性能：' + '\n')
    TestADPT12_Feature = out('./datasets/TestADP.txt')
    TestADPT12 = testADPT12(TestADPT12_Feature)
    TestADP_best(TestADPT12)