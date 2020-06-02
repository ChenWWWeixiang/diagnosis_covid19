import xlrd,os
import SimpleITK as sitk
import numpy as np
import sklearn.metrics as metric
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
res=np.load('../re/reader_cap_vs_covid.npy')
pre=res[:,-3:-1].astype(np.float)
pre=pre/pre.sum(1,keepdims=True)
pre=pre[:,1]
gt=res[:,-1].astype(np.float)
acc=np.sum(gt==(pre>0.5))
name=res[:,0]
print(acc)
plt.figure()
axes = plt.subplot(111)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Distinguishing COVID from CAP')

fpr,tpr,threshold = metric.roc_curve(gt, pre)
plt.plot(fpr, tpr,axes=axes)
axins = inset_axes(axes, width=2, height=1.5, loc='lower right',borderpad=2)
axins.plot(fpr, tpr)
#airesult=open('sum_scores.txt','r')
#airesult=airesult.readlines()
#air_name=np.array([ne.split('\t')[0] for ne in airesult])
#air_pre=[float(ne.split('\t')[1]) for ne in airesult]
#air_gt=[1-int(an.split('/')[-1][0]=='c' or  an.split('/')[-3]=='LIDC' or an.split('/')[-3]=='reader_ex') for an in air_name]
air_error=np.array(gt)!=(np.array(pre)>0.5)
air_error_name=name[air_error]
READER_E=np.zeros(100)
answer = open('../key_result/list/reader_cap_vs_covid.list', 'r')
answer=answer.readlines()
answer=[da.split(',')[0] for da in answer]
gt=[1-int('cap' in an.split(',')[0]) for an in answer]
answer=np.array(answer)
for id in range(1,6):
    workbook=xlrd.open_workbook("reader0"+str(id)+".xlsx")
    worksheet=workbook.sheet_by_index(0)
    #name=worksheet.col_values(0,3)
    pre=worksheet.col_values(3,2)
    for i,item in enumerate(pre):
        try:
            pre[i]=int(item)
            if pre[i]==2:
                pre[i]=0.5
        except:
            pre[i]=0.5
    pre=np.array(pre)
    gt=np.array(gt)
    READER_E+=(gt!=(pre>0.5))
    reader_error_name=answer[gt!=(pre>0.5)]
    inter=set(reader_error_name.tolist()).intersection(set(air_error_name.tolist()))
    upright = set(reader_error_name.tolist()).difference(set(air_error_name.tolist()))
    downleft = set(air_error_name.tolist()).difference(set(reader_error_name.tolist()))
    print([['',len(upright)],[len(downleft), len(inter)]])
    y=np.sum((gt==1)*(pre>0.5))/np.sum(gt==1)
    x=np.sum((gt==0)*(pre>0.5))/np.sum(gt==0)
    specifciity = np.sum((gt == 0) * (pre > 0.5)) / np.sum(gt == 0)
    sensitivity = np.sum((gt == 1) * (pre > 0.5)) / np.sum(gt == 1)
    ACC=pre==gt

    axins.scatter(specifciity,sensitivity)
    axes.scatter(specifciity,sensitivity)
    print(sensitivity,specifciity,np.sum(ACC))
plt.xlim([0.0,0.18])
plt.ylim([0.75,0.98])
axes.legend(['AI system','reader 1','reader 2','reader 3','reader 4','reader 5'],loc='lower left')
plt.savefig('jpgs/roc_cap_covid.jpg',bbox_inches='tight')

#plt.show()
xx=np.where(READER_E>=1)
yy=np.where(READER_E==0)
reader_error_name=answer[xx]
print(reader_error_name)

all_wrong_but_ai_right = set(reader_error_name.tolist()).difference(set(air_error_name.tolist()))
print('all_wrong_but_ai_right',len(list(all_wrong_but_ai_right)))
read_ok_name=answer[yy]
all_right_but_ai_wrong=set(read_ok_name.tolist()).intersection(set(air_error_name.tolist()))
print('all_right_but_ai_wrong',len(list(all_right_but_ai_wrong)))
all_wrong_and_ai_wrong=set(reader_error_name.tolist()).intersection(set(air_error_name.tolist()))
print('all_wrong_and_ai_wrong',len(list(all_wrong_and_ai_wrong)))
a=1

'''
import cv2
os.makedirs('/mnt/data9/all_wrong_but_ai_right/',exist_ok=True)
os.makedirs('/mnt/data9/all_right_but_ai_wrong/',exist_ok=True)

for id,item in enumerate(list(all_right_but_ai_wrong)):
    data = sitk.ReadImage(item)
    cls=item.split('/')[-2]
    V=sitk.GetArrayFromImage(data)
    L=sitk.ReadImage(item.replace(cls+'/',cls+'/'+cls+'_').replace('_data/','_segs/lungs/'))
    L=sitk.GetArrayFromImage(L)
    idx=np.where(L.sum(1).sum(1)>500)
    ii,jj,zz=np.where(L>0)
    V = V[idx[0], jj.min():jj.max(), zz.min():zz.max()]
    for idx, i in enumerate(range(0,V.shape[0],5)):
        data=V[i,:,:]
        data[data>700]=700
        data[data<-1200]=-1200
        data=data*255.0/1900
        data=data-data.min()
        data = data.astype(np.uint8)
        cv2.imwrite(os.path.join('/mnt/data9/all_right_but_ai_wrong/',cls+
                                 '_'+item.split('/')[-1]+'_'+str(i)+'.jpg'),data)

for id,item in enumerate(list(all_wrong_but_ai_right)):
    data = sitk.ReadImage(item)
    cls=item.split('/')[-2]
    V=sitk.GetArrayFromImage(data)
    L=sitk.ReadImage(item.replace(cls+'/',cls+'/'+cls+'_').replace('_data/','_segs/lungs/'))
    L=sitk.GetArrayFromImage(L)
    idx=np.where(L.sum(1).sum(1)>500)
    ii,jj,zz=np.where(L>0)
    V = V[idx[0], jj.min():jj.max(), zz.min():zz.max()]
    for idx, i in enumerate(range(0,V.shape[0],5)):
        data=V[i,:,:]
        data[data>700]=700
        data[data<-1200]=-1200
        data=data*255.0/1900
        data=data-data.min()
        data = data.astype(np.uint8)
        cv2.imwrite(os.path.join('/mnt/data9/all_wrong_but_ai_right/',item.split('/')[-2]+
                                 '_'+item.split('/')[-1]+'_'+str(id)+'_'+str(i)+'.jpg'),data)
'''
