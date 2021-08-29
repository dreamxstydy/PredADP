

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
f = plt.figure(figsize=(15,4))
import numpy as np
x = np.arange(-40,50,10)
fea = pd.read_csv('./ADP/Train/Features.csv',header=None)
fea = fea.iloc[1:,1:]
data = pd.read_csv('./ADP/Train/train.csv',header=None)
AAC = data.iloc[1:,1:21]
BPNC = data.iloc[1:,21:221]
CTD = data.iloc[1:,221:368]
DPC = data.iloc[1:,368:768]
x_feature_vectorAAC = TSNE().fit_transform(AAC)
x_feature_vectorBPNC = TSNE().fit_transform(BPNC)
x_feature_vectorCTD = TSNE().fit_transform(CTD)
x_feature_vectorDPC = TSNE().fit_transform(DPC)
x_feature_vector= TSNE().fit_transform(fea)
samples_count = DPC.shape[0]

adp_non = ['null']*samples_count
for i in range(814):
    if i < 407:
        adp_non[i]='ADP'
    else:
        adp_non[i] = 'Non-ADP'
adp_non = pd.Categorical(adp_non)
feature_vector_pdAAC = pd.DataFrame({'Dim1':x_feature_vectorAAC[:,0],'Dim2':x_feature_vectorAAC[:,1],'Category':adp_non})

feature_vector_pdBPNC = pd.DataFrame({'Dim1':x_feature_vectorBPNC[:,0],'Dim2':x_feature_vectorBPNC[:,1],'Category':adp_non})

feature_vector_pdCTD = pd.DataFrame({'Dim1':x_feature_vectorCTD[:,0],'Dim2':x_feature_vectorCTD[:,1],'Category':adp_non})

feature_vector_pdDPC = pd.DataFrame({'Dim1':x_feature_vectorDPC[:,0],'Dim2':x_feature_vectorDPC[:,1],'Category':adp_non})

feature_vector_pd = pd.DataFrame({'Dim1':x_feature_vector[:,0],'Dim2':x_feature_vector[:,1],'Category':adp_non})
f.add_subplot(1,5,1)
plt.title('AAC',fontsize=18)
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pdAAC,style="Category")
plt.xticks(x)
plt.yticks(x)
plt.text(-55,43,'(a)',fontdict={'size':14})
plt.legend(loc='upper right')

f.add_subplot(1,5,2)
plt.title('BPNC',fontsize=18)
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pdBPNC,style="Category")
plt.xticks(x)
plt.yticks(x)
plt.text(-55,43,'(b)',fontdict={'size':14})

plt.legend('',frameon=False)
f.add_subplot(1,5,3)
plt.title('CTD',fontsize=18)
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pdCTD,style="Category")
plt.xticks(x)
plt.yticks(x)
plt.text(-55,43,'(c)',fontdict={'size':14})
plt.legend('',frameon=False)
f.add_subplot(1,5,4)
plt.title('DPC',fontsize=18)
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pdDPC,style="Category")
plt.xticks(x)
plt.yticks(x)
plt.text(-55,43,'(d)',fontdict={'size':14})
plt.legend('',frameon=False)
f.add_subplot(1,5,5)
plt.title('4DFV',fontsize=18)
sns.scatterplot(x="Dim1", y="Dim2",
            hue="Category",
            data=feature_vector_pd,style="Category")
plt.xticks(x)
plt.yticks(x)
plt.text(-55,43,'(e)',fontdict={'size':14})
plt.legend('',frameon=False)

plt.savefig(f"ADP/Figure/tsne.png",dpi=800,format="png")
plt.show()


# feaadp = pd.read_csv('./ADPT12/Train/ADPT12Stack.csv',header=None)
# feaadp = feaadp.iloc[1:,1:]
# data = pd.read_csv('./ADPT12/Train/train.csv',header=None)
# ADPT12AAC = data.iloc[1:,1:21]
# ADPT12BPNC = data.iloc[1:,21:221]
# ADPT12CTD = data.iloc[1:,221:368]
# ADPT12DPC = data.iloc[1:,368:768]
#
# x_feature_vectorADPT12AAC = TSNE().fit_transform(ADPT12AAC)
# x_feature_vectorADPT12BPNC = TSNE().fit_transform(ADPT12BPNC)
# x_feature_vectorADPT12CTD = TSNE().fit_transform(ADPT12CTD)
# x_feature_vectorADPT12DPC = TSNE().fit_transform(ADPT12DPC)
# x_feature_vectorADPT12 = TSNE().fit_transform(feaadp)
#
#
# samples_count = ADPT12DPC.shape[0]
# adp_non = ['null']*samples_count
# for i in range(407):
#     if i < 248:
#         adp_non[i]='ADPT1'
#     else:
#         adp_non[i] = 'ADPT2'
# adp_non = pd.Categorical(adp_non)
# feature_vector_pdADPT12AAC = pd.DataFrame({'Dim1':x_feature_vectorADPT12AAC[:,0],'Dim2':x_feature_vectorADPT12AAC[:,1],'Category':adp_non})
#
# feature_vector_pdADPT12BPNC = pd.DataFrame({'Dim1':x_feature_vectorADPT12BPNC[:,0],'Dim2':x_feature_vectorADPT12BPNC[:,1],'Category':adp_non})
#
# feature_vector_pdADPT12CTD = pd.DataFrame({'Dim1':x_feature_vectorADPT12CTD[:,0],'Dim2':x_feature_vectorADPT12CTD[:,1],'Category':adp_non})
#
# feature_vector_pdADPT12DPC = pd.DataFrame({'Dim1':x_feature_vectorADPT12DPC[:,0],'Dim2':x_feature_vectorADPT12DPC[:,1],'Category':adp_non})
#
# feature_vector_pdADPT12 = pd.DataFrame({'Dim1':x_feature_vectorADPT12[:,0],'Dim2':x_feature_vectorADPT12[:,1],'Category':adp_non})
#
# text_font3={
#
#     'style':'normal',
#     'weight':'normal',
#       'color':'k',
#       'size':14}
#
#
# text_font4={
#     #'family':'Times New Roman',
#     'style':'normal',
#     'weight':'normal',
#       'color':'k',
#       'size':12,
#        }
#
# # plt.title('',fontdict=text_font3)
# f.add_subplot(1,5,1)
# plt.title('AAC',fontsize=18)
# sns.scatterplot(x="Dim1", y="Dim2",
#             hue="Category",
#             data=feature_vector_pdADPT12AAC,
# style="Category")
# plt.xticks(x)
# plt.yticks(x)
# plt.text(-55,43,'(a)',fontdict={'size':14})
# plt.legend(loc='upper right')
# f.add_subplot(1,5,2)
# plt.title('BPNC',fontsize=18)
# sns.scatterplot(x="Dim1", y="Dim2",
#             hue="Category",
#             data=feature_vector_pdADPT12BPNC,style="Category")
# plt.xticks(x)
# plt.yticks(x)
# plt.text(-55,43,'(b)',fontdict={'size':14})
# plt.legend('',frameon=False)
#
# f.add_subplot(1,5,3)
# plt.title('CTD',fontsize=18)
# sns.scatterplot(x="Dim1", y="Dim2",
#             hue="Category",
#             data=feature_vector_pdADPT12CTD,style="Category")
# plt.xticks(x)
# plt.yticks(x)
# plt.text(-55,43,'(c)',fontdict={'size':14})
# plt.legend('',frameon=False)
#
# f.add_subplot(1,5,4)
# plt.title('DPC',fontsize=18)
# sns.scatterplot(x="Dim1", y="Dim2",
#             hue="Category",
#             data=feature_vector_pdADPT12DPC,style="Category")
# plt.xticks(x)
# plt.yticks(x)
# plt.text(-55,43,'(d)',fontdict={'size':14})
# plt.legend('',frameon=False)
# f.add_subplot(1,5,5)
# plt.title('4DFV',fontsize=18)
# sns.scatterplot(x="Dim1", y="Dim2",
#             hue="Category",
#             data=feature_vector_pdADPT12,style="Category")
# plt.xticks(x)
# plt.yticks(x)
# plt.text(-55,43,'(e)',fontdict={'size':14})
# plt.legend('',frameon=False)
#
# plt.savefig(f"ADPT12/Figure/tsne.eps",dpi=300,format="eps")
# plt.show()






