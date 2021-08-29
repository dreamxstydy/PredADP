import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
index = np.arange(5)
data2 = np.array([0.975,0.963,0.972,0.970,0.977])
plt.rcParams['font.sans-serif']=['SimHei']#正常显示中文汉字
bar_width = 0.3
fig=plt.figure(figsize=(6,5))
plt.bar(index, data2, width=0.2, color='y',label='Dataset-2')
plt.xlabel('特征组合')
plt.ylabel('AUC')
plt.ylim(0.95,0.98)
plt.xticks([0,1,2,3,4],['ALL','ALL-AAC','ALL-BPNC','ALL-CTD','ALL-DPC'])
Path("./Plot1").mkdir(exist_ok=True)
plt.savefig(f"./Plot2/TrainADP12.eps",dpi=300,format="eps")
plt.legend(loc='upper right')
plt.show()

