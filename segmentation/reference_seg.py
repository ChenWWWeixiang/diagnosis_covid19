import SimpleITK as sitk 
import numpy as np
#from segmentation.lungmask import mask
import glob
from tqdm import tqdm
import os 
from segmentation.predict import predict,get_model
#from segmentation.unet import UNet
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
lung_dir = '/mnt/data11/seg_of_XCT/lung/CAP/'
leision_dir = '/mnt/data11/seg_of_XCT/lesion/CAP/'
root_dir = '/home/cwx/extra/dr_ct_data/CT/CAP'
filelist = glob.glob(root_dir)
os.makedirs(leision_dir,exist_ok=True)
model2 = './checkpoint_final.pth'
model = get_model(model2,n_classes=2)
print('get model done')
for filepath in filelist:
    imagelist = glob.glob(filepath+'/*.nii')
    for imagepath in tqdm(imagelist, dynamic_ncols=True):
        imagename = imagepath.split('/')[-1]
        batch_id = imagepath.split('/')[-2]
        if os.path.exists(leision_dir+batch_id+'_'+imagename.replace('.nii','_label.nrrd')):
            print(imagename)
            continue
        input_image = sitk.ReadImage(imagepath)
        segmentation = predict(input_image, model = model,batch_size=16,lesion=True)
        segmentation[segmentation>1]=1
        lung_image = sitk.ReadImage(lung_dir+batch_id+'_'+imagename)
        lung_data = sitk.GetArrayFromImage(lung_image)
        leision_seg = lung_data*segmentation
        leision_seg=np.array(leision_seg,np.uint8)
        result_out= sitk.GetImageFromArray(leision_seg)
        result_out.CopyInformation(input_image)
 
        sitk.WriteImage(result_out,leision_dir+batch_id+'_'+imagename.replace('.nii','_label.nrrd'))
        print(imagename)
