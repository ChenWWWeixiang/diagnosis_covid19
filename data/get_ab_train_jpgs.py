import SimpleITK as sitk
import numpy as np
import random
from PIL import Image
import cv2,os,glob
#input_path='/home/cwx/extra/CAP'
#input_mask='/mnt/data6/CAP/resampled_seg'

output_path_slices='/mnt/data9/m_leision_jpgs_train/'
os.makedirs(output_path_slices,exist_ok=True)
cnt=0
train_list='trainlist_ct_only.list'
train_list=open(train_list,'r').readlines()
train_list=[da for da in train_list if '/covid/' in da]
dd=['lung_1st','lung_2rd']
for idx,name in enumerate(train_list):
    set_name=name.split('/')[-2]
    input_path = '/mnt/data6/lung_data/'
    input_mask = '/mnt/data6/lung_seg2/'
    input_lesion_mask = '/mnt/data9/m_leision_seg/'
    set_id = int(name.split(',')[0].split('/')[-1].split('_')[0])
    if  set_id>2:
        continue
    lesion_name =  glob.glob(input_lesion_mask+name.split(',')[0].split('/')[-1].split('_')[1]+'_*.nii')
    if len(lesion_name)==0 :
        continue
    cnt+=1
    continue
    for alesion in lesion_name:
        if set_id==2:
            inner_id=int(alesion.split('/')[-1].split('_')[0])-100
            phase_id=alesion.split('/')[-1].split('_')[1].split('-mask')[0]
        else:
            inner_id = int(alesion.split('/')[-1].split('_')[0])
            phase_id = alesion.split('/')[-1].split('_')[1].split('-mask')[0]
        volume = sitk.ReadImage(os.path.join(input_path,dd[set_id-1],str(inner_id)+'_'+phase_id+'.nii'))
        mask = sitk.ReadImage(os.path.join(input_mask,'illPatient'+str(set_id)+'_'+str(inner_id)+'_'+phase_id+'_label.nii'))
        L = sitk.ReadImage(alesion)
        L = sitk.GetArrayFromImage(L)
        L[L>0]=1
        M=sitk.GetArrayFromImage(mask)
        M[M>0]=1
        V = sitk.GetArrayFromImage(volume)

        sums = M.sum(1).sum(1)
        idd=np.where(sums>500)
        iddx=np.where(M>0)
        M = M[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
        V = V[idd[0],iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
        L = L[idd[0], iddx[1].min():iddx[1].max(),iddx[2].min():iddx[2].max()]
        sums2 = L.sum(1).sum(1)
        sums2=np.where(sums2>50)[0]
        for idx, i in enumerate(range(0,V.shape[0],3)):
            data=V[i,:,:]
            data[data>500]=500
            data[data<-1200]=-1200
            data=data*255.0/1700
            data=data-data.min()
            data=np.stack([data,M[i,:,:]*data,M[i,:,:]*255],-1)#mask one channel
            data = data.astype(np.uint8)
            if i in sums2:

                cv2.imwrite(os.path.join(output_path_slices,'abnor'+'_'
                                         +alesion.split('/')[-1].split('.nii')[0]
                                         +'_'+str(int(i/(V.shape[0])*100))+'.jpg'),data)
            else:
                cv2.imwrite(os.path.join(output_path_slices,'nor'+'_'
                                         +alesion.split('/')[-1].split('.nii')[0]
                                         +'_'+str(int(i/(V.shape[0])*100))+'.jpg'),data)
print(cnt)
