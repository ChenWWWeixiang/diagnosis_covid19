import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.model_selection import cross_val_score
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, df, cls, scoring="neg_mean_squared_error", cv = 3))
    return(rmse)
sns.set()
#sns.load_dataset("flights")
#df = sns.load_dataset("iris")
# Load the brain networks example dataset
df = pd.read_csv("R_withfake_features.csv",error_bad_lines=False)
cls=df.pop('label')
id=df.pop('id')
#M=df.max()._values
#m=df.min()._values
#np.save('M.npy',M)
#np.save('m.npy',m)
#df=(df - df.min()) / (df.max() - df.min())
#id_small=id[0:id.size:20]
lut = dict(zip(cls.unique(), "rbg"))
row_colors = cls.map(lut)

sns.clustermap(df, row_colors=row_colors,standard_scale=1,figsize=(20, 20))
plt.title('Cluster Heatmap of Features Before LASSO',x=10,y=1,fontsize=20)
plt.savefig("BeforeCHM.jpg")
corr = df.corr()
f,ax= plt.subplots(figsize = (15, 15),nrows=1)
# cubehelix mapÑÕÉ«/home/cwx/extra/MVI-R
sns.heatmap(corr, ax = ax, vmax=1, vmin=-1)
ax.set_title('Corr Matrix Before LASSO',fontsize=20)
f.savefig('corr_heatmap_before.jpg', bbox_inches='tight')
#plt.show()


model = Lasso(alpha=0.6)
model.fit(df._values, cls._values)
coef = pd.Series(model.coef_, index = df.columns)
coef[coef.abs()<1e-4]=0
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#print(rmse_cv(model).mean())

after_df = df.iloc[:,(coef!=0)._values]
# matplotlib colormap
f,ax= plt.subplots(figsize = (7, 7),nrows=1)
sns.heatmap(after_df.corr(), ax = ax, vmax=1, vmin=-1,)
ax.set_title('Corr Matrix After LASSO',fontsize=20)
f.savefig('corr_heatmap_after.jpg', bbox_inches='tight')
#plt.show()
sns.clustermap(after_df, row_colors=row_colors,standard_scale=1,figsize=(10, 20))
plt.title('Cluster Heatmap of Features After LASSO',x=10,y=1,fontsize=20)
plt.savefig("After_CHM.jpg")
imp_coef = coef[coef!=0]
#imp_coef.sort()
name=np.where(coef!=0)[0]
co=imp_coef.values
saving=np.save('coefs.npy',np.stack([name,co],-1))
#matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
plt.figure(figsize=(10,6))
imp_coef.plot(kind = "barh",fontsize=7)
print(imp_coef.keys())
plt.subplots_adjust(left=0.5, wspace=0.2, hspace=0.2)
#print(imp_coef.values)
plt.axis('tight')
plt.title("Coefficients in the Lasso Model",fontsize=20)
plt.savefig("CIM.jpg", bbox_inches='tight')



#plt.show()
