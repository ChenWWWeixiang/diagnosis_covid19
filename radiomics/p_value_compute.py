import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
fn='in'

df = pd.read_csv('r_features.csv',error_bad_lines=False)
cls=df.pop('label').astype(int)
name=df.pop('id')

model = joblib.load("train_model_"+fn+".m")
coef = pd.Series(model.coef_, index = df.columns)
imp_coef = pd.concat([coef.sort_values().head(3),
                     coef.sort_values().tail(3)])
DIS=df[imp_coef.index]

if fn=='cap':
    d1 = DIS[cls == 1]
else:
    d1 = DIS[cls == 0]
d2=DIS[cls==2]
for i in range(6):
    ks=stats.ks_2samp(d2.values[:,i], d1.values[:,i]).pvalue
    t=stats.ttest_ind(d2.values[:,i], d1.values[:,i]).pvalue
    print(d2.keys()[i]+','+str(imp_coef[i])+','+str(ks)+','+str(t))
score=(DIS*imp_coef).sum(1)
if fn=='cap':
    d1 = score[cls == 1]
else:
    d1 = score[cls == 0]
d2=score[cls==2]
ks=stats.ks_2samp(d2.values, d1.values).pvalue
t=stats.ttest_ind(d2.values, d1.values).pvalue
print('R-Score,-,'+str(ks)+','+str(t))

