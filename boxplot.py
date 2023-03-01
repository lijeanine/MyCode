import pandas as pd
import numpy as np
import seaborn as sns
AddMLP=np.load('./AddMLP/AddMLP.npy')
parallel = np.load('./parallel/parallel.npy')
serial = np.load('./serial/serial.npy')
Unit3Model = np.load('./Unit3Model/Unit3Model.npy')

df = pd.DataFrame({'CSF1':AddMLP[:,1],'CSF2':parallel[:,1],'CSF3':serial[:,1],'CSF4':Unit3Model[:,1],
                   'CGM1':AddMLP[:,2],'CGM2':parallel[:,2],'CGM3':serial[:,2],'CGM4':Unit3Model[:,2],
                   'WM1':AddMLP[:,3],'WM2':parallel[:,3],'WM3':serial[:,3],'WM4':Unit3Model[:,3],
                   'VT1':AddMLP[:,5],'VT2':parallel[:,5],'VT3':serial[:,5],'VT4':Unit3Model[:,5],
                   'CB1':AddMLP[:,6],'CB2':parallel[:,6],'CB3':serial[:,6],'CB4':Unit3Model[:,6],
                   'DGM1':AddMLP[:,7],'DGM2':parallel[:,7],'DGM3':serial[:,7],'DGM4':Unit3Model[:,7],
                   'BS1':AddMLP[:,8],'BS2':parallel[:,8],'BS3':serial[:,8],'BS4':Unit3Model[:,8],
                   'HA1':AddMLP[:,9],'HA2':parallel[:,9],'HA3':serial[:,9],'HA4':Unit3Model[:,9]})
df["id"] = df.index
df = pd.wide_to_long(df, stubnames=['CSF','CGM','WM','VT','CB','DGM','BS','HA'], i=['id'], j='Model').reset_index().drop('id', axis=1)
df = df.melt(id_vars='Model',var_name='tissue',value_name='DSC')
sns.boxplot(data=df, x='tissue', y='DSC', hue='Model',linewidth=2,fliersize=0,saturation=1)
