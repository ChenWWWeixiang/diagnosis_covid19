import numpy as np
from sklearn.manifold import TSNE
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2,os
x = np.load('../saves/X.npy').squeeze(1)
c=np.load('../saves/Y.npy')

#ts = TSNE(n_components=2)

#ts.fit_transform(x)
plt.figure(figsize=(10,10))
#y = ts.fit_transform(x)
y=np.load('../saves/T_X.npy')
z=np.load('../saves/Z.npy')
#np.save('../saves/T_X.npy',y)

w=2000
h=2000
hugemap=np.ones((w,h,3))*255
posmap=np.zeros((w,h,3))
def draw_margin(I,color,line_width):
    for i in range(I.shape[1]):
        I[:line_width,  i, :] = color
        I[-line_width:, i, :] = color
    for i in range(I.shape[0]):
        I[i,:line_width, :] = color
        I[i,-line_width:, :] = color
    return I
def loadimg(path):
    data=sitk.ReadImage(path)
    data=sitk.GetArrayFromImage(data)
    L=path.replace('_data','_segs/lungs')
    dirname=path.split('/')[5]
    lung=sitk.ReadImage(L.replace(dirname+'/',dirname+'/'+dirname+'_'))
    lung=sitk.GetArrayFromImage(lung)
    idx,idy,idz=np.where(lung>0)
    data=data[idx.min():idx.max(),idy.min():idy.max(),idz.min():idz.max()]
    lens=data.shape[0]
    data=data[lens//2,:,:]
    data[data > 500] = 500
    data[data < -1200] = -1200
    data = data * 255.0 / 1700
    data = data - data.min()
    return data
Color=[np.array([0,0,255]),np.array([0,255,0]),np.array([255,0,0]),np.array([255,255,0])]

for i in range(4):
    plt.scatter(y[c==i, 0], y[c==i, 1])

if False:
    for i in range(y.shape[0]):
        #plt.scatter(y[c==i, 0], y[c==i, 1])
        cx=int(y[i, 0]*w/160)+1000
        cy=int(y[i, 1]*h/160)+1000
        if np.sum(posmap[cx-30:cx+30,cy-30:cy+30,:])==0:
            I=loadimg(z[i][0])
            I=np.stack([I,I,I],-1).astype(np.uint8)
            I=cv2.resize(I,(150,150))
            I=draw_margin(I, Color[c[i]], 3)
            hugemap[cx - 75:cx + 75, cy - 75:cy + 75,:] = I
            posmap[cx - 75:cx + 75, cy - 75:cy + 75, :] = 1
    cv2.imwrite('jpgs/img_tsne.jpg',hugemap)

for i in range(y.shape[0]):
    if y[i,0]>40 and c[i]==3:
        I=loadimg(z[i][0])
        cv2.imwrite('group/right_'+z[i][0].split('/')[-1]+'.jpg',I)
    if y[i,1]>40 and c[i]==3:
        I=loadimg(z[i][0])
        cv2.imwrite('group/up_'+z[i][0].split('/')[-1]+'.jpg',I)
    if y[i,1]>20 and c[i]==3 and y[i,0]<-20:
        I=loadimg(z[i][0])
        cv2.imwrite('group/left'+z[i][0].split('/')[-1]+'.jpg',I)
    if c[i]==2:
        I = loadimg(z[i][0])
        cv2.imwrite('group/influanza' + z[i][0].split('/')[-1]+'.jpg', I)
    if c[i]==1:
        I = loadimg(z[i][0])
        cv2.imwrite('group/cap' + z[i][0].split('/')[-1]+'.jpg', I)
    if c[i]==0:
        I = loadimg(z[i][0])
        cv2.imwrite('group/healthy' + z[i][0].split('/')[-1]+'.jpg', I)
plt.title('t-SNE Curve', fontsize=14)
plt.legend(['Healthy','CAP','Influenza','COVID19'])
plt.savefig('jpgs/tSNE.jpg')
plt.show()
