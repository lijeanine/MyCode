import pandas as pd
import numpy as np
import seaborn as sns
AddMLP=np.load('./AddMLP/AddMLP.npy')
parallel = np.load('./parallel/parallel.npy')
serial = np.load('./serial/serial.npy')
Unit3Model = np.load('./Unit3Model/Unit3Model.npy')

df = pd.DataFrame({'A1':AddMLP[:,1],'A2':parallel[:,1],'A3':serial[:,1],'A4':Unit3Model[:,1],
                   'B1':AddMLP[:,2],'B2':parallel[:,2],'B3':serial[:,2],'B4':Unit3Model[:,2],
                   'C1':AddMLP[:,3],'C2':parallel[:,3],'C3':serial[:,3],'C4':Unit3Model[:,3],
                   'D1':AddMLP[:,5],'D2':parallel[:,5],'D3':serial[:,5],'D4':Unit3Model[:,5],
                   'E1':AddMLP[:,6],'E2':parallel[:,6],'E3':serial[:,6],'E4':Unit3Model[:,6],
                   'F1':AddMLP[:,7],'F2':parallel[:,7],'F3':serial[:,7],'F4':Unit3Model[:,7],
                   'G1':AddMLP[:,8],'G2':parallel[:,8],'G3':serial[:,8],'G4':Unit3Model[:,8],
                   'H1':AddMLP[:,9],'H2':parallel[:,9],'H3':serial[:,9],'H4':Unit3Model[:,9]})
df["id"] = df.index
df = pd.wide_to_long(df, stubnames=['A','B','C','D','E','F','G','H'], i=['id'], j='Model').reset_index().drop('id', axis=1)
df = df.melt(id_vars='Model',var_name='tissue',value_name='DSC')
sns.boxplot(data=df, x='tissue', y='DSC', hue='Model',linewidth=2,fliersize=0,saturation=1)
