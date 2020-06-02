import pandas as pd
import csv
import sklearn.cluster as cluster
import numpy as np
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt

name=[
    'label','original_glszm_GrayLevelVariance',
'log-sigma-1-0-mm-3D_firstorder_Minimum',
'log-sigma-3-0-mm-3D_firstorder_Median',
'log-sigma-3-0-mm-3D_glszm_HighGrayLevelZoneEmphasis',
'log-sigma-3-0-mm-3D_glszm_SmallAreaHighGrayLevelEmphasis',
'log-sigma-3-0-mm-3D_ngtdm_Complexity',
'log-sigma-5-0-mm-3D_glrlm_HighGrayLevelRunEmphasis',
'wavelet-LH_firstorder_90Percentile',
'wavelet-HL_firstorder_10Percentile',
'wavelet-HH_glcm_Autocorrelation',
'wavelet-LL_glrlm_GrayLevelNonUniformity',
'wavelet-LL_glszm_GrayLevelNonUniformity',
'wavelet-LL_gldm_SmallDependenceHighGrayLevelEmphasis',

]
name=['label','log-sigma-1-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis',
       'log-sigma-3-0-mm-3D_glszm_LargeAreaHighGrayLevelEmphasis',
       'log-sigma-5-0-mm-3D_firstorder_Energy',
       'log-sigma-5-0-mm-3D_firstorder_TotalEnergy',
       'wavelet-LH_firstorder_Energy', 'wavelet-HL_firstorder_Energy',
       'wavelet-HL_firstorder_TotalEnergy', 'wavelet-LL_firstorder_Energy',
       'wavelet-LL_firstorder_TotalEnergy',
       'wavelet-LL_glcm_ClusterProminence']
beta=[0.00523709,0.00682345,0.03613166,0.00517925,-0.01565185,
0.00721222 ,-0.00670309 ,0.01241168 ,-0.00922436 ,0.01110218 ,
-0.00577271 ,-0.00657399 ,0.00519171 ,
]
aaa=np.load('coefs.npy')
beta=aaa[:,1]
name=['id','label','original_glrlm_RunLengthNonUniformity',
       'original_glszm_LargeAreaEmphasis',
       'log-sigma-1-0-mm-3D_glszm_LargeAreaLowGrayLevelEmphasis',
       'log-sigma-1-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
       'log-sigma-3-0-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
       'wavelet-LH_firstorder_Variance', 'wavelet-HL_glcm_ClusterProminence',
       'wavelet-HL_ngtdm_Complexity',
       'wavelet-HH_glrlm_RunLengthNonUniformity',
       'wavelet-HH_gldm_DependenceNonUniformity',
       'wavelet-LL_firstorder_Maximum',
       'wavelet-LL_glrlm_GrayLevelNonUniformity',
       'wavelet-LL_glrlm_LongRunHighGrayLevelEmphasis',
       'wavelet-LL_glrlm_ShortRunHighGrayLevelEmphasis',
       'wavelet-LL_ngtdm_Complexity']
plt.style.use('ggplot')
with open('R_withfake_features.csv','r') as f:
    df = pd.read_csv(f)
    df=df[name]
    df.to_csv('15_left.csv')
    score=(df.iloc[:,1:]*np.array(beta)).sum(1)
    score.name='score'
    df=pd.concat([df, score], axis=1)
    #df = (df - df.mean()) / df.std()
    #df.pop('id')
    #cls=df.pop('label')
    df1 = df.iloc[np.where(df['label']==1)[0],:]
    df0 = df.iloc[np.where(df['label'] == 0)[0], :]
    #df1.pop('label')
    df0.pop('label')
    df1.pop('label')
    #df = (df - df.mean()) / df.std()
    #cls.name='label'
    #df=pd.concat([df,cls],axis=1)
    #
    df1 = (df1 - df.mean()) / df.std()
    df0 = (df0 - df.mean()) / df.std()
    #df1.to_csv('df1.csv')
    #df0.to_csv('df0.csv')
    m_1=df1.mean(0).values
    m_0=df0.mean(0).values
    s_1=df1.std(0).values
    s_0=df0.std(0).values
    c=[m_1+s_1,m_1-s_1,m_0+s_0,m_0-s_0]

    #fig, axes = plt.subplots()

    #sns.violinplot(data=df,hue='label',x=name[2])
    #plt.legend(['1','2','s','2a'])
    #plt.ylim([-1,2])
    plt.figure(figsize=(10,10))
    plt.plot(m_1,color='r',label='lesion')
    plt.plot(c[0], linestyle='dotted',color='r')
    plt.plot(c[1], linestyle='dotted', color='r')
    plt.plot(m_0,color='b',label='normal')
    plt.plot(c[2], linestyle='dotted',color='b')
    plt.plot(c[3], linestyle='dotted', color='b')
    plt.legend()
    #plt.xlabel(name[1:])
    x = np.linspace(0, 15, 16)
    plt.xticks(x, list(name[1:]+['score']), rotation=90, fontsize=10)
    plt.subplots_adjust(left=0.05, wspace=0.2, hspace=0.2,
                        bottom=0.49, top=0.94)
    plt.title('Feature Distributions After LASSO for Diffenrent Classes')
    plt.savefig('f_d.jpg')
    plt.show()
    a=1


