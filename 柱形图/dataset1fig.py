import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
index = np.arange(5)
data1 = np.array([0.812,0.808,0.787,0.797,0.798])
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文汉字
bar_width = 0.3
fig=plt.figure(figsize=(6,5))
plt.bar(index, data1, width=0.2, color='y',label='Dataset-1')
plt.xlabel('特征组合')
plt.ylabel('AUC')
plt.ylim(0.75,0.85)
plt.xticks([0,1,2,3,4],['ALL','ALL-AAC','ALL-BPNC','ALL-CTD','ALL-DPC'])
Path("./Plot1").mkdir(exist_ok=True)
plt.savefig(f"./Plot1/TrainADP.eps",dpi=300,format="eps")
plt.legend(loc='upper right')
plt.show()

