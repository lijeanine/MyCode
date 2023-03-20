import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
AddMLP=np.load('./AddMLP/AddMLP.npy')
parallel = np.load('./parallel/parallel.npy')
serial = np.load('./serial/serial.npy')
Unit3Model = np.load('./Unit3Model/Unit3Model.npy')

df = pd.DataFrame({'CSF_BFANet-PT':parallel[:,1],'CSF_BFANet-ST':serial[:,1],'CSF_BFANet':Unit3Model[:,1],
                   'CGM_BFANet-PT':parallel[:,2],'CGM_BFANet-ST':serial[:,2],'CGM_BFANet':Unit3Model[:,2],
                   'WM_BFANet-PT':parallel[:,3],'WM_BFANet-ST':serial[:,3],'WM_BFANet':Unit3Model[:,3],
                   'VT_BFANet-PT':parallel[:,5],'VT_BFANet-ST':serial[:,5],'VT_BFANet':Unit3Model[:,5],
                   'CB_BFANet-PT':parallel[:,6],'CB_BFANet-ST':serial[:,6],'CB_BFANet':Unit3Model[:,6],
                   'DGM_BFANet-PT':parallel[:,7],'DGM_BFANet-ST':serial[:,7],'DGM_BFANet':Unit3Model[:,7],
                   'BS_BFANet-PT':parallel[:,8],'BS_BFANet-ST':serial[:,8],'BS_BFANet':Unit3Model[:,8],
                   'HA_BFANet-PT':parallel[:,9],'HA_BFANet-ST':serial[:,9],'HA_BFANet':Unit3Model[:,9]})
df["id"] = df.index
df = pd.wide_to_long(df, stubnames=['CSF','CGM','WM','VT','CB','DGM','BS','HA'], i=['id'], j='方法',sep='_',suffix='\D+').reset_index().drop('id', axis=1)
df = df.melt(id_vars='方法',var_name='待分割组织',value_name='DSC')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示符号的问题
#sns.set(font='SimHei', font_scale=0.8)        # 解决Seaborn中文显示问题

plt.figure(dpi=200, figsize=(8, 4))
ax=sns.boxplot(data=df, x='待分割组织', y='DSC', hue='方法',linewidth=1,fliersize=0,saturation=1,width=0.5)
