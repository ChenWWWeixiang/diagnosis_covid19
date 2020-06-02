import os
import SimpleITK as sitk
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
age_list='raw_ages.txt'
reload=False
inpuath='/mnt/data7/NCP_mp_CTs/crop/lesions'
def showing(df,name,title):
    plt.figure(figsize=(7,15))
    plt.subplot(5,1,1)
    idx=df[(df['sex']==1)].index
    sns.distplot(df['tend_cls'][idx],label='male',kde=False,bins=np.arange(-0.5,8.5,1),hist=True,norm_hist=True)
    idx=df[(df['sex']==0)].index
    sns.distplot(df['tend_cls'][idx],label='female',kde=False,bins=np.arange(-0.5,8.5,1),hist=True,norm_hist=True)
    plt.legend()
    #plt.ylim([0,1])
    plt.xlim([-0.5,8.5])
    plt.xlabel('')
    plt.title('all')
    for age in range(1,5):
        plt.subplot(5,1,age+1)
        idx=df[(df['age'] ==age * (df['sex']==1))].index
        sns.distplot(df['tend_cls'][idx],label='male',kde=False,bins=np.arange(-0.5,8.5,1),hist=True,norm_hist=True)
        idx=df[(df['age'] ==age *(df['sex']==0))].index
        sns.distplot(df['tend_cls'][idx],label='female',kde=False,bins=np.arange(-0.5,8.5,1),hist=True,norm_hist=True)
        plt.xlim([-0.5, 8.5])
        #plt.ylim([0, 1])
        plt.xlabel('')
        plt.legend()
        plt.title(str(age*20))

    plt.xlabel('Tendency Class')
    #plt.legend()
    plt.tight_layout()
    plt.suptitle(title)
    plt.subplots_adjust(top=0.90)
    plt.savefig(name)
    plt.show()
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
                Lsize.append([this_name,stage.split('.mha')[0],lesion_size,this_age,this_sex])

    Lsize=np.array(Lsize)
    df = pd.DataFrame(Lsize, columns=('name','time','size','age','sex'))
    df.to_csv('all_infos.csv',index=False)
else:
    df=pd.read_csv('all_infos.csv')
cluster_cnt=pd.DataFrame(columns=('name','age','sex','tend_cls'))
for a_person in df['name'].unique():
    this_df = df[(df['name']==a_person)]
    this_df=this_df.sort_values(by=["time"], ascending=[True])
    if this_df.shape[0]<3:
        continue
    this_3p=this_df.iloc[:3]
    sizes=this_3p['size']._values
    up1=(sizes[1]+1e-5)/(sizes[0]+1e-5)
    up2=(sizes[2]+1e-5)/(sizes[1]+1e-5)
    if up1<0.8:
        a=0
    elif up1>1.2:
        a=2
    else:
        a=1
    if up2<0.8:
        b=0
    elif up2>1.2:
        b=2
    else:
        b=1
    cls=a*3+b
    ss={'name':a_person,'age':this_3p['age']._values[0],'sex':this_3p['sex']._values[0],'tend_cls':cls}
    cluster_cnt=cluster_cnt.append(ss,ignore_index=True)
showing(cluster_cnt,'cls_agesex_3p.jpg','Distribution of Tendency')