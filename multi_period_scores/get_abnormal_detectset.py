import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import cv2,os
#input_path='/home/cwx/extra/CAP'
#input_mask='/mnt/data6/CAP/resampled_seg'
set_name='abnormal'
input_path='/home/cwx/extra/covid_project_data/covid'
input_mask='/home/cwx/extra/covid_project_segs/lungs/covid'
input_lesion_mask='/home/cwx/extra/covid_project_segs/lesion/covid'

output_path_pos='/mnt/data9/covid_detector_jpgs/pos_'+set_name
output_path_neg='/mnt/data9/covid_detector_jpgs/neg_'+set_name

os.makedirs(output_path_pos,exist_ok=True)
os.makedirs(output_path_neg,exist_ok=True)

cnt=0

name_list=os.listdir(input_path)

reader = sitk.ImageSeriesReader()
for idx,name in enumerate(name_list):
    set_id=int(name.split('_')[0])
    if set_id>3:
        continue
    volume = sitk.ReadImage(os.path.join(input_path,name))
    mask_name =  'covid_'+name

    lesion_name =  'covid_' + name.split('.nii')[0]+'_label.nrrd'
    L = sitk.ReadImage(os.path.join(input_lesion_mask, lesion_name))
    L = sitk.GetArrayFromImage(L)
    L[L>0]=1
    mask=sitk.ReadImage(os.path.join(input_mask,mask_name))

    M=sitk.GetArrayFromImage(mask)
    V = sitk.GetArrayFromImage(volume)
    #M = M[:V.shape[0], :, :]

    sums_all = M.sum(1).sum(1)
    idd=np.where(sums_all>500)[0]
    M=M[idd,:,:]
    V=V[idd,:,:]
    L=L[idd,:,:]
    sums_pos = L.sum(1).sum(1)
    idd_pos = np.where(sums_pos > 50)[0]
    for idx, i in enumerate(range(1,V.shape[0]-3,3)):
        data=V[i-1:i+2,:,:]
        data[data>700]=700
        data[data<-1200]=-1200
        data=data*255.0/1900

        data=data-data.min()

        data=np.concatenate([data[0:2,:,:],M[i:i+1,:,:]*255],0)#mask one channel
        data = data.astype(np.uint8).transpose(1,2,0)

        if i in idd_pos:
            cv2.imwrite(os.path.join(output_path_pos,name.split('.n')[0]+'_'+str(i)+'.jpg'),data)
        else:
            cv2.imwrite(os.path.join(output_path_neg,name.split('.n')[0]+'_'+str(i)+'.jpg'),data)
