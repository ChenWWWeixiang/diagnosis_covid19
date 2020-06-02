import SimpleITK as sitk
import os,cv2
import numpy as np
from scipy.spatial.distance import cdist
o_img_nii = '/mnt/data9/cam/pre'
o_msk_nii = '/mnt/data9/cam/mask'
o_lung_nii = '/mnt/data9/cam/lung'
f=open('distance.txt','w')
f.writelines('name'+','+'value'+'\n')
for item in os.listdir(o_lung_nii):
    try:
        lung=sitk.ReadImage(os.path.join(o_lung_nii,item))
        lung=sitk.GetArrayFromImage(lung)*255
        #lung=np.stack([lung,lung,lung],0)
        side=cv2.Canny(lung.astype(np.uint8), 200, 300)
        side_points=np.array(np.where(side==255))
        #cv2.imwrite('temp.jpg',canny)
        mask=sitk.ReadImage(os.path.join(o_msk_nii,item))
        mask=sitk.GetArrayFromImage(mask)
        index=np.where(mask>0)
        centroid=np.mean(index,1)
        dist = cdist(centroid[np.newaxis,:], side_points.transpose(), 'euclidean')
        dist=dist.min()
        f.writelines(item+','+str(dist)+'\n')
        print(item,dist)
    except:
        continue
f.close()
