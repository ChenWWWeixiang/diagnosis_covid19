import os
import numpy as np
import matplotlib.pyplot as plt
f1=open('distance.txt','r')
f2=open('HFD.txt','r')
f3=open('HFD3D.txt','r')

distances=f1.readlines()
distances=[float(dis.split(',')[-1]) for dis in distances]
#distances=np.array(distances)
#distances=distances/distances.sum()

n=plt.hist(distances,80)
plt.close()
plt.figure()
x=n[1]
m=x.min()
M=x.max()
x = np.arange(m, M, (M - m) / 80)
n=n[0]
n=n/n.sum()
plt.bar(x, n, align='center',width=(M-m)/80)
plt.xlabel('Distance (pixel)')
plt.ylabel('Probability')
plt.title('Distribution of Distance Feature of Regions of Attention')
plt.savefig('../ipt_results/dis.jpg')

plt.figure()
hfd=f2.readlines()
hfd=[float(dis.split(',')[-1]) for dis in hfd]
#hfd=np.array(hfd)
#hfd=hfd/hfd.sum()
n=plt.hist(hfd,80)
plt.close()
plt.figure()
x=n[1]
m=x.min()
M=x.max()
x = np.arange(m, M, (M - m) / 80)
n=n[0]
n=n/n.sum()
plt.bar(x, n, align='center',width=(M-m)/80)
plt.xlabel('Hausdorff fractal dimension')
plt.ylabel('Probability')
plt.title('Distribution of Fractal Dimension for Regions of Attention')
plt.savefig('../ipt_results/hfd.jpg')

plt.figure()
hfd=f3.readlines()
hfd=[float(dis.split(',')[-1]) for dis in hfd]
#hfd=np.array(hfd)
#hfd=hfd/hfd.sum()
n=plt.hist(hfd,80)
plt.close()
plt.figure()
x=n[1]
m=x.min()
M=x.max()
x = np.arange(m, M, (M - m) / 80)
n=n[0]
n=n/n.sum()
plt.bar(x, n, align='center',width=(M-m)/80)
plt.xlabel('Hausdorff fractal dimension')
plt.ylabel('Probability')
plt.title('Distribution of Fractal Dimension for Gray Level Mesh of Attention Area ')
plt.savefig('../ipt_results/hfd3d.jpg')
