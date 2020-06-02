import SimpleITK as sitk
import os,glob
root='/home/cwx/baidunetdiskdownload/天池大赛肺部结节智能诊断'
output='/mnt/data11/extra_data/healthy2'
for adir in os.listdir(root):
    all_data=glob.glob(os.path.join(root,adir,'*.mhd'))
    for item in all_data:
        data=sitk.ReadImage(item)
        sitk.WriteImage(data,os.path.join(output,'2_'+item.split('/')[-1].replace('.mhd','_19900101_-1_N.nii')))
