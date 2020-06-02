import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import cv2,os
#input_path='/home/cwx/extra/CAP'
#input_mask='/mnt/data6/CAP/resampled_seg'

output_path_slices='/mnt/data9/covid_detector_jpgs/selected_train_pos/nor'
os.makedirs(output_path_slices,exist_ok=True)
cnt=0
train_list='trainlist_xct.list'
train_list2='trainlist_ct_only.list'
train_list3='train_ex.list'
train_list=open(train_list,'r').readlines()+open(train_list2,'r').readlines()+open(train_list3,'r').readlines()

for idx,name in enumerate(train_list):
    set_name=name.split('/')[-2]
    if not 'healthy' in set_name:
        continue
    #if not 'cap'in  set_name:
    #    continue#wait segmentation
    #input_path = '/home/cwx/extra/covid_project_data/' + set_name
    #input_mask = '/home/cwx/extra/covid_project_segs/lungs/' + set_name
    #input_path = '/home/cwx/extra/dr_ct_data/CT/' + set_name
    #input_mask = '/mnt/data11/seg_of_XCT/lung/' + set_name
    #input_lesion_mask='/home/cwx/extra/covid_project_segs/lesion/' + set_name
    volume = sitk.ReadImage(name.split(',')[0])
    mask = sitk.ReadImage(name.split(',')[1][:-1])

    M=sitk.GetArrayFromImage(mask)
    M[M>0]=1
    V = sitk.GetArrayFromImage(volume)

    sums = M.sum(1).sum(1)
    idd=np.where(sums>500)
    iddx=np.where(M>0)
    M = M[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
    V = V[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]

    if not 'healthy' in set_name  and False:
        L = L[idd[0], iddx[1].min():iddx[1].max(), iddx[2].min():iddx[2].max()]
        sums2 = L.sum(1).sum(1)
        sums2=np.where(sums2>50)[0]

    for idx, i in enumerate(range(0,V.shape[0],10)):
        if not 'healthy' in set_name and False:
            if not i in sums2:
                continue
        data=V[i,:,:]
        data[data>500]=500
        data[data<-1200]=-1200#-1200~500
        data=data*255.0/1700
        data=data-data.min()

        data=np.stack([data,M[i,:,:]*data,M[i,:,:]*255],-1)#mask one channel
        data = data.astype(np.uint8)

        cv2.imwrite(os.path.join(output_path_slices,'nor_'+set_name+'_'
                                 +name.split(',')[0].split('/')[-1].split('.nii')[0]
                                 +'_'+str(int(i/(V.shape[0])*100))+'.jpg'),data)
