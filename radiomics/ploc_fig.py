import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.model_selection import cross_val_score,train_test_split,KFold
import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
model = joblib.load("train_model.m")
df = pd.read_csv('new_r_features.csv',error_bad_lines=False)
cls=df.pop('label').astype(int)

id=df.pop('id')
coef = pd.Series(model.coef_, index = df.columns)

#saving=np.save('coefs.npy',np.stack([name,co],-1))
#matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
plt.figure()
imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(5)])
imp_coef.plot(kind = "barh")
print(imp_coef.keys())

plt.subplots_adjust(left = 0.4)
#print(imp_coef.values)
plt.axis('tight')
plt.title("Coefficients in the Lasso Model")
plt.savefig("coffs.jpg", bbox_inches='tight')
joblib.dump(model, "train_model.m")