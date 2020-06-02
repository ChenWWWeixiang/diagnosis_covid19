import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
filepath='../all_ages_genders.txt'
datas=open(filepath,'r').readlines()
tips=pd.read_csv('age_sex.csv')
tips["sex"] = pd.Categorical(tips["gender"], tips["gender"].unique())
def getthings(datas):
    name=[da.split('\t')[0] for da in datas]
    age=[int(da.split('\t')[1]) for da in datas]
    sex=[da.split('\t')[-1][:-1]=='M' for da in datas]

    return name,age,sex

cap=[na for na in datas if 'CAP' in na]
control=[na for na in datas if 'c' in na]
ild=[na for na in datas if 'ILD' in na]
covid=[na for na in datas if not 'ILD' in na and not 'c' in na and not 'CP' in na]
a_name,a_age,a_sex=getthings(datas)
c_name,c_age,c_sex=getthings(control)
cap_name,cap_age,cap_sex=getthings(cap)
i_name,i_age,i_sex=getthings(ild)
covid_name,covid_age,covid_sex=getthings(covid)
#plt.subplot(2,2,1)
df=pd.DataFrame(np.array([a_age,a_sex]).transpose(),columns=['age', 'gender'])
df["gender"] = pd.Categorical(df["gender"], df["gender"].unique())
sns.violinplot(data=df,y='age',hue='gender',split=True,label='all',palette="muted")
#plt.subplot(2,2,2)
#plt.legend()
#plt.show()

df=pd.DataFrame(np.array([c_age,c_sex]).transpose(),columns=['age', 'gender'])
df["gender"] = pd.Categorical(df["gender"], ["M", "F"])
sns.violinplot(data=df,y='age',hue='gender',kde=True,label='control')
#plt.subplot(2,2,3)
df=pd.DataFrame(np.array([cap_age,cap_sex]).transpose(),columns=['age', 'gender'])
df["gender"] = pd.Categorical(df["gender"], ["M", "F"])
sns.violinplot(data=df,y='age',hue='gender', kde=True,label='cap')
#plt.subplot(2,2,4)
df=pd.DataFrame(np.array([i_age,i_sex]).transpose(),columns=['age', 'gender'])
df["gender"] = pd.Categorical(df["gender"], ["M", "F"])
sns.violinplot(data=df,y='age',hue='gender',kde=True,label='ild')

df=pd.DataFrame(np.array([covid_age,covid_sex]).transpose(),columns=['age', 'gender'])
df["gender"] = pd.Categorical(df["gender"], ["M", "F"])
sns.violinplot(data=df,y='age',hue='gender',kde=True,label='covid')


plt.legend()
plt.show()