import os
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
CLS=['healthy','cap','HxNx','covid']
def get_CI(value,res):
    sorted_scores=np.array(value)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    res.append(str(np.mean(value)) + ' (' + str(confidence_lower) + '-' + str(confidence_upper) + ')')
    return res
def plot_a_group(s,gt,pre):
    AUC = []
    ACC = []
    REC = []
    PRE = []
    SAUC = []
    y_one_hot = label_binarize(gt, np.arange(4))
    norm_x = pre / pre.max(axis=0)
    for i in range(200):
        train_x, test_x, train_y, test_y = train_test_split(pre, y_one_hot, test_size=0.2)
        train_x = train_x / train_x.max(axis=0)
        auc = metric.roc_auc_score(train_y, train_x, average='micro')
        AUC.append(auc)

        prediction = np.argmax(train_x, 1)
        groundtruth = np.argmax(train_y, 1)
        prediction[np.max(train_x[:, 1:], 1) > 0.80] = np.argmax(train_x[np.max(train_x[:, 1:], 1) > 0.80, 1:], 1) + 1
        ACC.append(np.mean(prediction == groundtruth))
        recall = []
        precision = []
        sauc = []
        for cls in range(4):
            recall.append(np.sum((prediction == cls) * (groundtruth == cls)) / (np.sum(groundtruth == cls)+1e-5))
            precision.append(np.sum((prediction == cls) * (groundtruth == cls)) / (np.sum(prediction == cls)+1e-5))
            sauc.append(metric.roc_auc_score(train_y[cls, :], train_x[cls, :]))
        SAUC.append(sauc)
        REC.append(recall)
        PRE.append(precision)
    PRE = np.array(PRE)
    REC = np.array(REC)
    SAUC = np.array(SAUC)
    Res = [s]
    Res = get_CI(AUC, Res)
    Res = get_CI(ACC, Res)
    Res = get_CI(SAUC[:, 0], Res)
    Res = get_CI(REC[:, 0], Res)
    Res = get_CI(PRE[:, 0], Res)
    Res = get_CI(SAUC[:, 1], Res)
    Res = get_CI(REC[:, 1], Res)
    Res = get_CI(PRE[:, 1], Res)
    Res = get_CI(SAUC[:, 2], Res)
    Res = get_CI(REC[:, 2], Res)
    Res = get_CI(PRE[:, 2], Res)
    Res = get_CI(SAUC[:, 3], Res)
    Res = get_CI(REC[:, 3], Res)
    Res = get_CI(PRE[:, 3], Res)
    f.writerow(Res)
    BAR_plot.append(np.mean(SAUC, 0))
    #plt.figure(1)
    # fpr,tpr,threshold = metric.roc_curve(y_one_hot, norm_x)
    #fpr, tpr, thresholds = metric.roc_curve(y_one_hot.ravel(), norm_x.ravel())
    #plt.plot(fpr, tpr, label=s + ', AUC={:.2f}'.format(metric.auc(fpr, tpr)))
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--ress", help="A list of npy files which record the performance.",
                    default=['../key_result/test.npy'])
parser.add_argument("-o", "--output_file", help="Output file path", type=str,
                    default='../saves/results_subset.csv')
args = parser.parse_args()

#res=np.load('ipt_results/results/train.npy')
if isinstance(args.ress,str):
    ress=eval(args.ress)
else:
    ress=args.ress
BAR_plot=[]
with open(args.output_file,'w') as f:
    f=csv.writer(f)
    f.writerow(['name', 'all-AUC', 'all-ACC', 'healthy-AUC', 'healthy-recall', 'healthy-precision',
                'CAP-AUC', 'CAP-recall', 'CAP-precision', 'HxNx-AUC', 'HxNx-recall', 'HxNx-precision',
                'COVID-AUC', 'COVID-recall', 'COVID-precision'])
    for a_res in ress:
        res = np.load(a_res)
        if res.shape[1]==5:
            pre=np.array(res[:,:-1],np.float)
            gt=np.array(res[:,-1],np.float)
        else:
            name=res[:, 0]
            pre = np.array(res[:, 1:-1], np.float)
            gt = np.array(res[:, -1], np.float)
        ## gender subset
        sex=[int(item[-5]=='M') for item in name]
        sex=np.array(sex)
        for sexi in range(2):
            AUC = []
            ACC = []
            REC = []
            PRE = []
            SAUC=[]
            y_one_hot = label_binarize(gt[sex==sexi], np.arange(4))
            norm_x=pre[sex==sexi,:]/ pre[sex==sexi,:].max(axis=0)
            print(sexi, np.sum(sex == sexi))
            for i in range(200):
                train_x, test_x, train_y, test_y = train_test_split(pre[sex==sexi,:], y_one_hot, test_size=0.2)
                train_x=train_x/train_x.max(axis=0)
                auc = metric.roc_auc_score(train_y, train_x, average='micro')
                AUC.append(auc)

                prediction = np.argmax(train_x, 1)
                groundtruth = np.argmax(train_y, 1)
                prediction[np.max(train_x[:, 1:], 1) > 0.80] = np.argmax(train_x[np.max(train_x[:, 1:], 1) > 0.80, 1:],
                                                                         1) + 1
                ACC.append(np.mean(prediction == groundtruth))
                recall = []
                precision = []
                sauc = []
                for cls in range(4):
                    recall.append(np.sum((prediction == cls) * (groundtruth == cls)) / np.sum(groundtruth == cls))
                    precision.append(np.sum((prediction == cls) * (groundtruth == cls)) / np.sum(prediction == cls))
                    sauc.append(metric.roc_auc_score(train_y[cls, :], train_x[cls, :]))
                SAUC.append(sauc)
                REC.append(recall)
                PRE.append(precision)
            PRE = np.array(PRE)
            REC = np.array(REC)
            SAUC = np.array(SAUC)
            if sexi==1:
                s='Male'
            else:
                s='Female'
            Res=[s]
            Res = get_CI(ACC, Res)
            Res = get_CI(SAUC[:, 0], Res)
            Res = get_CI(REC[:, 0], Res)
            Res = get_CI(PRE[:, 0], Res)
            Res = get_CI(SAUC[:, 1], Res)
            Res = get_CI(REC[:, 1], Res)
            Res = get_CI(PRE[:, 1], Res)
            Res = get_CI(SAUC[:, 2], Res)
            Res = get_CI(REC[:, 2], Res)
            Res = get_CI(PRE[:, 2], Res)
            Res = get_CI(SAUC[:, 3], Res)
            Res = get_CI(REC[:, 3], Res)
            Res = get_CI(PRE[:, 3], Res)
            f.writerow(Res)
            BAR_plot.append(np.mean(SAUC,0))
            #plt.figure(1)
            #fpr,tpr,threshold = metric.roc_curve(y_one_hot, norm_x)
            #fpr, tpr, thresholds = metric.roc_curve(y_one_hot.ravel(), norm_x.ravel())

            #plt.plot(fpr, tpr, label=s+', AUC={:.2f}'.format(metric.auc(fpr, tpr)))

        ## age subset
        age = [int(item.split('_')[-2]) // 20 for item in name]
        age = np.array(age)
        for agei in range(1, 4):
            AUC = []
            ACC = []
            REC = []
            PRE = []
            SAUC = []
            y_one_hot = label_binarize(gt[age == agei], np.arange(4))
            norm_x = pre[age == agei, :] / pre[age == agei, :].max(axis=0)
            print(agei,np.sum(age==agei))
            for i in range(200):
                train_x, test_x, train_y, test_y = train_test_split(pre[age==agei,:], y_one_hot, test_size=0.2)
                train_x=train_x/train_x.max(axis=0)
                auc=metric.roc_auc_score(train_y,train_x,average='micro')
                AUC.append(auc)

                prediction=np.argmax(train_x,1)
                groundtruth=np.argmax(train_y,1)
                prediction[np.max(train_x[:,1:],1) > 0.80]=np.argmax(train_x[np.max(train_x[:,1:],1) > 0.80,1:],1) + 1
                ACC.append(np.mean(prediction==groundtruth))
                recall=[]
                precision=[]
                sauc=[]
                for cls in range(4):
                    recall.append(np.sum((prediction==cls)*(groundtruth==cls))/(np.sum(groundtruth==cls)+1e-5))
                    precision.append(np.sum((prediction==cls)*(groundtruth==cls))/(np.sum(prediction==cls)+1e-5))
                    sauc.append(metric.roc_auc_score(train_y[cls,:], train_x[cls,:]))
                SAUC.append(sauc)
                REC.append(recall)
                PRE.append(precision)
            PRE=np.array(PRE)
            REC=np.array(REC)
            SAUC=np.array(SAUC)
            s="{:d}-{:d}".format(agei*20+1,agei*20+20)
            Res=[s]
            Res=get_CI(AUC,Res)
            Res = get_CI(ACC, Res)
            Res = get_CI(SAUC[:, 0], Res)
            Res = get_CI(REC[:,0], Res)
            Res = get_CI(PRE[:,0], Res)
            Res = get_CI(SAUC[:, 1], Res)
            Res = get_CI(REC[:,1], Res)
            Res = get_CI(PRE[:,1], Res)
            Res = get_CI(SAUC[:, 2], Res)
            Res = get_CI(REC[:,2], Res)
            Res = get_CI(PRE[:,2], Res)
            Res = get_CI(SAUC[:, 3], Res)
            Res = get_CI(REC[:,3], Res)
            Res = get_CI(PRE[:,3], Res)
            f.writerow(Res)
            BAR_plot.append(np.mean(SAUC, 0))
            #plt.figure(1)
            #fpr,tpr,threshold = metric.roc_curve(y_one_hot, norm_x)
            #fpr, tpr, thresholds = metric.roc_curve(y_one_hot.ravel(), norm_x.ravel())
            #plt.plot(fpr, tpr, label=s+', AUC={:.2f}'.format(metric.auc(fpr, tpr)))

        ##stage subset
        time = [int(item.split('_')[-3][-4:]) for item in name]
        time = np.array(time)
        person = [item.split('/')[-2] + '/' + item.split('/')[-1].split('_')[0] + '_' +
                  item.split('/')[-1].split('_')[1] for item in name]
        #for idx, pe in enumerate(person):
        #    if pe.split('/')[0] == 'HxNx':
        #        person[idx] = pe.split('/')[0] + '/' + pe.split('/')[1].split('_')[0]
        unit_person = list(set(person))
        STAGEI = []
        STAGEII = []
        STAGE_MEAN = []
        person = np.array(person)
        unit_person = np.array(unit_person)
        cnt2 = 0
        for iperson in unit_person:
            this_idx = np.where(person == iperson)[0]
            this_time = time[this_idx]
            if len(this_time) >= 2:
                cnt2 += 1
            else:
                continue
            sorted_idx = this_idx[np.argsort(this_time)]
            STAGEI.append([pre[sorted_idx[0], :], gt[sorted_idx[0]]])
            STAGEII.append([pre[sorted_idx[1], :], gt[sorted_idx[1]]])
            STAGE_MEAN.append([np.mean(pre[sorted_idx,:], 0), gt[sorted_idx[1]]])
        print(cnt2)
        STAGEI = np.array(STAGEI)
        STAGEII = np.array(STAGEII)
        STAGE_MEAN = np.array(STAGE_MEAN)
        plot_a_group('stage I', np.array(STAGEI[:, 1], np.int), np.stack(STAGEI[:, 0]))
        plot_a_group('stage II', np.array(STAGEII[:, 1], np.int), np.stack(STAGEII[:, 0]))
        plot_a_group('stage mix', np.array(STAGE_MEAN[:, 1], np.int), np.stack(STAGE_MEAN[:, 0]))

plt.figure(2)
BAR_plot=np.array(BAR_plot)
data=pd.DataFrame(BAR_plot,columns=['Non-pneumonia','CAP','Influenza','COVID-19'])

data.plot.bar()
plt.ylim([0.85,1])
plt.xticks((0,1,2,3,4,5,6,7,8),['female','male','21-40','41-60','61-80','Stage I','Stage II','Stage Mix'])
plt.xticks(rotation=45)
plt.title('AUC for Different Sub-Sets')
plt.ylabel('AUC')
plt.xlabel('Subset')
plt.tight_layout()
#plt.xlim([0,4.5])
plt.legend(loc=4)
plt.savefig('jpgs/bar_auc_gender.jpg')
#plt.show()
