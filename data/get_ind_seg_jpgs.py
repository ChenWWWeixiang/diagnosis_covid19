import cv2,os,glob
import SimpleITK as sitk
import numpy as np
segs='/mnt/data9/independent_segs/lungs'
raw='/mnt/data9/independent_data'
out_path='/mnt/data9/independent_crop'
for type in os.listdir(segs):
    for volume in os.listdir(os.path.join(segs,type)):
        person=volume.split('_')[1]
        stage = volume.split('_')[2]
        R=sitk.ReadImage(os.path.join(raw,type,person+'_'+stage))
        R=sitk.GetArrayFromImage(R)
        M=sitk.ReadImage(os.path.join(segs,type,volume))
        M=sitk.GetArrayFromImage(M)
        for i in range(M.shape[0]):
            m = M[i,:,:]
            I = R[i, :, :]
            I=(I+1400)/1500*255
            IMG=np.stack([I,I,m*255],-1).astype(np.uint8)
            #yy,xx=np.where(I>0)
            #try:
            #    I=I[yy.min():yy.max(),xx.min():xx.max()]
            #except:
            #    a=1
            name=os.path.join(out_path,volume.split('.')[0]+'_'+str(i)+'.jpg')
            cv2.imwrite(name,IMG)
            a=1