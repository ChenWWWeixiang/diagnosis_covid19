import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
H3D='HFD3D.txt'
HFD='HFD.txt'
DIS='distance.txt'

def func(DIS,fn):
    DIS=open(DIS,'r').readlines()
    DIS=DIS[1:]
    distances=[float(dis.split(',')[-1][:-1]) for dis in DIS]
    name=[dis.split(',')[0] for dis in DIS]
    cls=[]
    distances=np.array(distances)
    for iname in name:
        if 'cap' in iname:
            cls.append(1)
        elif 'AB-in' in iname:
            cls.append(2)
        else:
            cls.append(3)
    cls=np.array(cls)
    d3=distances[cls==3]
    d1=distances[cls==1]
    d2=distances[cls==2]

    pval1=stats.ks_2samp(d3, d1,mode='auto').pvalue
    print(fn,'CAP',pval1,stats.ttest_ind(d3, d1).pvalue)
    pval2=stats.ks_2samp(d3,d2,mode='auto').pvalue
    print(fn,'Influenza',pval2,stats.ttest_ind(d3, d2).pvalue)
    sns.distplot(d3,label='COVID-19',norm_hist=True)
    sns.distplot(d2,label='Inluenza',norm_hist=True)
    sns.distplot(d1,label='CAP',norm_hist=True,)
    plt.xlim([distances.min(),distances.max()])
    plt.legend()
    plt.title(fn+' distribution of attention region')
    plt.savefig(fn+'_dis.jpg')
    #plt.show()
    plt.close()

func(DIS,'Distance')
func(HFD,'2D fractal dimension')
func(H3D,'3D fractal dimension')