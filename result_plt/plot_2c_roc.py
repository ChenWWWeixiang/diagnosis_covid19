#plot 2c roc
import os
import csv
import numpy as np
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
saving='jpgs/reader_healthy_vs_ill.jpg'
parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['../key_result/reader_healthy_vs_ill.npy'])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='csvs/reader_healthy_vs_ill.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name','AUC','ACC','Specificity','Sensitivity','PPV','NPV','F1','Youden'])
    for a_res in ress:
        res = np.load(a_res,allow_pickle=True)
        if res.shape[1]==3:
            pre=np.array(res[:,1],np.float)
            gt=np.array(res[:,2],np.float)
        else:
            pre = np.array(res[:, -2],np.float)
            gt = np.array(res[:, -1], np.float)
        AUC=[]
        ACC=[]
        TNR=[]
        TPR=[]
        PPV=[]
        NPV=[]
        F1=[]
        YOUDEN=[]
        for i in range(200):
            train_x, test_x, train_y, test_y = train_test_split(pre, gt, test_size=0.2)
            prediction = train_x
            auc=metric.roc_auc_score(train_y,train_x)
            AUC.append(auc)
            p=0.50000001
            ACC.append(metric.accuracy_score(prediction>p,train_y))
            TPR.append(np.sum((train_y == 1) * (prediction > p)) / np.sum(train_y == 1))
            TNR.append(np.sum((train_y == 0) * (prediction < p)) / np.sum(train_y == 0))
            PPV.append(metric.precision_score(train_y,prediction>p))
            NPV.append(np.sum((train_y == 0) * (prediction < p)) / np.sum(prediction < p))
            F1.append(metric.f1_score(train_y,prediction>p))
            YOUDEN.append(np.sum((train_y == 1) * (prediction > p)) / np.sum(train_y == 1)+
                          np.sum((train_y == 0) * (prediction < p)) / np.sum(train_y == 0)-1)
        Res=[a_res.split('/')[-1].split('.npy')[0]]
        Res=get_CI(AUC,Res)
        Res = get_CI(ACC, Res)
        Res = get_CI(TPR, Res)
        Res = get_CI(TNR, Res)
        Res = get_CI(PPV, Res)
        Res = get_CI(NPV, Res)
        Res = get_CI(F1, Res)
        Res = get_CI(YOUDEN, Res)
        f.writerow(Res)

        plt.figure(1)
        fpr,tpr,threshold = metric.roc_curve(gt, pre)
        plt.plot(fpr, tpr, label=a_res.split('/')[-1].split('.npy')[0]+' AUC={:.4f}'.format(metric.auc(fpr, tpr)))
       # plt.figure(2)
        #precision, recall, t = metric.precision_recall_curve(gt, pre)
        #plt.plot(recall, precision,label=a_res.split('/')[-1].split('.npy')[0])

plt.figure(1)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
#plt.show()
plt.savefig(saving)

