import os
import SimpleITK as sitk
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
age_list='raw_ages.txt'
reload=False
inpuath='/mnt/data7/NCP_mp_CTs/crop/lesions'
if reload:
    Lsize=[]
    df=pd.read_csv(age_list,sep='\t')
    for setname in os.listdir(inpuath):
        for person in os.listdir(os.path.join(inpuath,setname)):
            for stage in os.listdir(os.path.join(inpuath,setname,person)):
                data=sitk.ReadImage(os.path.join(inpuath,setname,person,stage))
                data=sitk.GetArrayFromImage(data)
                lesion_size=np.sum(data)/data.shape[0]/data.shape[1]/data.shape[2]
                this_name=setname+'/'+person.split('_')[0]
                idx=df[(df['name'] ==this_name)].index
                try:
                    this_age=int(df['age'][idx])//20
                    this_sex=int(df['sex'][idx]=='M')
                except:
                    this_age=45
                    this_sex=1
                Lsize.append([lesion_size,this_age,this_sex])
    Lsize=np.array(Lsize)
    df = pd.DataFrame(Lsize, columns=('size','age','sex'))
    df.to_csv('size_age.csv',index=False)
else:
    df=pd.read_csv('size_age.csv')

plt.figure(figsize=(7,15))
plt.subplot(5,1,1)
idx=df[(df['sex']==1)].index
sns.distplot(df['size'][idx]*100,kde=True,label='male',bins=np.arange(0,9,0.25),norm_hist=True)
idx=df[(df['sex']==0)].index
sns.distplot(df['size'][idx]*100,kde=True,label='female',bins=np.arange(0,9,0.25),norm_hist=True)
plt.legend()
plt.ylim([0,3])
plt.xlim([0,8])
plt.xlabel('')
plt.title('all')
for age in range(1,5):
    plt.subplot(5,1,age+1)
    idx=df[(df['age'] ==age * (df['sex']==1))].index
    sns.distplot(df['size'][idx]*100,kde=True,label='male',bins=np.arange(0,9,0.25),norm_hist=True)
    idx=df[(df['age'] ==age *(df['sex']==0))].index
    sns.distplot(df['size'][idx]*100,kde=True,label='female',bins=np.arange(0,9,0.25),norm_hist=True)
    plt.xlim([0,8])
    plt.ylim([0, 3])
    plt.xlabel('')
    plt.legend()
    plt.title(str(age*20))

plt.xlabel('Size (%)')
#plt.legend()

plt.tight_layout()
plt.suptitle('Distribution of Relative Size')
plt.subplots_adjust(top=0.90)
plt.savefig('dis_size_agesex.jpg')
plt.show()