import SimpleITK as sitk
import os
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
lungs_cap='/home/cwx/extra/covid_project_segs/lungs/cap'
l_cap='/home/cwx/extra/covid_project_segs/lesion/cap'
lungs_covid='/home/cwx/extra/covid_project_segs/lungs/covid'
l_covid='/home/cwx/extra/covid_project_segs/lesion/covid'
CAP=[]
reload=False
fig,ax=plt.subplots()

def show_bins(data,name,fig_name):
    bin_edges=np.array([0.0,0.0001,0.001,0.01,0.1,1])
    num_g=len(data)
    bin_width = 1
    x = np.arange(1,6,1)
    width = bin_width / num_g
    #plt.figure()
    for i in range(num_g):
        hist, _ = np.histogram(data[i],bins=bin_edges)
        hist=hist/hist.sum()
        plt.bar(x+i*width-width/2,hist,width,alpha = 0.9,label=name[i])
    plt.xlim([0,6])
    plt.xticks(np.arange(1,6,1),["<0.01%",'0.01%-0.1%','0.1%-1%','1%-10%','>10%'])
    plt.xlabel('Infected Size (lesion area/ lung area)')
    plt.ylabel('Pecentage')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

if reload:
    for item in os.listdir(l_cap):
        lesion=sitk.ReadImage(os.path.join(l_cap,item))
        lesion=sitk.GetArrayFromImage(lesion)
        lung_name=item.replace('_label.nrrd','.nii')
        lung=sitk.ReadImage(os.path.join(lungs_cap,lung_name))
        lung=sitk.GetArrayFromImage(lung)
        lung[lung>1]=1
        lesion[lesion>1]=1
        CAP.append(np.sum(lesion)/np.sum(lung))
    COVID=[]
    for item in os.listdir(l_covid):
        lesion=sitk.ReadImage(os.path.join(l_covid,item))
        lesion=sitk.GetArrayFromImage(lesion)
        lung_name=item.replace('_label.nrrd','.nii')
        lung=sitk.ReadImage(os.path.join(lungs_covid,lung_name))
        lung=sitk.GetArrayFromImage(lung)
        lung[lung>1]=1
        lesion[lesion>1]=1
        COVID.append(np.sum(lesion)/np.sum(lung))

    COVID=np.array(COVID)
    CAP=np.array(CAP)
    np.save('CAP.npy',CAP)
    np.save('COVID.npy',COVID)
else:
    CAP = np.load('CAP.npy')
    COVID = np.load('COVID.npy')
show_bins([CAP,COVID],['CAP','COVID'],'CAP_VS_COVID.jpg')
plt.show()