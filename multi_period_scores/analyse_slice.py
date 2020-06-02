#plot 2c roc
import os
import csv
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from sklearn.calibration import calibration_curve
def get_CI(value,res):
    sorted_scores=np.array(value)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    res.append(str(np.mean(value)) + ' (' + str(confidence_lower) + '-' + str(confidence_upper) + ')')
    return res
import argparse
parser = argparse.ArgumentParser()
saving='jpgs/roc_slice_locate.jpg'
parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['../re/slice_locate.npy'])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='slice.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
reload=True
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress

for a_res in ress:
    res = np.load(a_res,allow_pickle=True)
    pre = np.array(res[:, -2],np.float)
    gt = np.array(res[:, -1], np.float)
    names=np.array(res[:, 0])
    if reload:
        scan_name=[na.split('/')[-1] for na in names]
        scan_name = [na.split('nor_')[-1].split('.jpg')[0] for na in scan_name]
        zs = np.array([int(na.split('_')[-1]) for na in scan_name])
        scan_name=[na.split('_')[0]+'_'+na.split('_')[1]+'_'+na.split('_')[2]+'_'+na.split('_')[3]+'_'+na.split('_')[4] for na in scan_name]
        scan_name=list(set(scan_name))
        P=[]
        G=[]
        sigP=[]
        sigG=[]
        for aname in scan_name:
            idx=[id for id,na in enumerate(names) if aname in na]
            this_z=zs[idx]
            this_pre=pre[idx]
            this_gt=gt[idx]
            sorted_idx=np.argsort(this_z)
            this_pre=this_pre[sorted_idx]
            this_gt = this_gt[sorted_idx]
            this_pre=signal.medfilt(this_pre,5)
            P+=this_pre.tolist()
            sigP.append(this_pre)
            G+=this_gt.tolist()
            sigG.append(this_gt)
            a=1

        G=np.array(G)
        P = np.array(P)
        np.save('G.npy',G)
        np.save('P.npy', P)
        np.save('sigP.npy',sigP)
        np.save('sigG.npy', sigG)
    else:
        G=np.load('G.npy',allow_pickle=True)
        P=np.load('P.npy',allow_pickle=True)
        sigP=np.load('sigP.npy',allow_pickle=True)
        sigG=np.load('sigG.npy',allow_pickle=True)
    plt.figure(1)
    fpr,tpr,threshold = metric.roc_curve(G, P)
    aa=0
    daa=0
    tt_dice=-1
    tt=-1
    for th in np.arange(0.1,1,0.01):
        da=0
        for thisp,thisg in zip(sigP,sigG):
            thisp=thisp>th
            dice=np.sum(thisg*thisp)*1.0/(np.sum(thisg+thisp)*1.0+1e-6)
            da+=dice
        da=da/len(sigP)
        if da>daa:
            daa=da
            tt_dice=th
        acc=metric.accuracy_score( G,P > th)
        if acc>aa:
            aa=acc
            tt=th
    print(tt,aa)
    print(tt_dice, daa)
    plt.plot(fpr, tpr, label=a_res.split('/')[-1].split('.npy')[0]+' AUC={:.4f}'.format(metric.roc_auc_score(G, P)))

plt.figure(1)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()
#plt.savefig(saving)

