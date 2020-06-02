import os
import SimpleITK as sitk
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import cv2
age_list='raw_ages.txt'
reload=True
inpuath='/mnt/data9/mp_NCPs/reg_to_one/lesions'
map=[100,100,100]
if reload:
    Lsize=[]
    df=pd.read_csv(age_list,sep='\t')
    for setname in os.listdir(inpuath):
        for person in os.listdir(os.path.join(inpuath,setname)):
            for stage in os.listdir(os.path.join(inpuath,setname,person)):
                data=sitk.ReadImage(os.path.join(inpuath,setname,person,stage))
                data=sitk.GetArrayFromImage(data)
                a=1


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

