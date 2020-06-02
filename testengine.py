from torch.autograd import Variable
import torch
import time,tqdm,shutil
import torch.optim as optim
from datetime import datetime, timedelta
#from data import LipreadingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import NCPJPGtestDataset,NCPJPGtestDataset_new,IndtestDataset
import os, cv2
import toml
from models.net2d import densenet121,densenet161,resnet152,resnet152_plus,resnet152_R,resnet50
import numpy as np
#from models.g_cam import GuidedPropo
import matplotlib as plt
KEEP_ALL=False
SAVE_DEEP=False
import argparse


def _validate(modelOutput, labels, length,topn=1):
    modelOutput=list(np.exp(modelOutput.cpu().numpy())[:length,-1])#for covid19
    #pos_count=np.sum(np.array(modelOutput)>0.5)

    modelOutput.sort()
    averageEnergies = np.mean(modelOutput[-topn:])
    iscorrect = labels.cpu().numpy()==(averageEnergies>0.5)
    pred=(averageEnergies>0.5)
    return averageEnergies,iscorrect,pred

def _validate_cp(modelOutput, labels, length,topn=1):
    averageEnergies = np.exp(modelOutput.cpu().numpy()[:length, :]).mean(0)
    pred = np.argmax(averageEnergies)
    iscorrect = labels.cpu().numpy() == pred
    return averageEnergies.tolist(), iscorrect, pred

def _validate_ind(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    modelOutput=np.exp(modelOutput.cpu().numpy())
    cppro=np.sum(modelOutput[:,1:3],1)
    healthypre=modelOutput[:,0]
    ncp_pre = modelOutput[:, -1]
    modelOutput=np.stack([healthypre,cppro,ncp_pre],-1)
    for i in range(0,modelOutput.shape[1]):
        t = modelOutput[:length, i].tolist() # for covid19
        t.sort()
        if i==0:
            averageEnergies.append(np.mean(t[-1:]))
        else:
            averageEnergies.append(np.mean(t[-topn:]))
    averageEnergies = averageEnergies / np.sum(averageEnergies, keepdims=True)
    pred=np.argmax(averageEnergies)
    if pred==2:
        pred=3
    label=labels.cpu().numpy()
    iscorrect = label == pred
    return averageEnergies.tolist(), [iscorrect],pred


def _validate_healthy_or_not(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    averageEnergies2=[]
    modelOutput=np.exp(modelOutput.cpu().numpy())
    illpro=np.mean(modelOutput[:,1:],1)
    healthypre=modelOutput[:,0]
    modelOutput=np.stack([healthypre,illpro],-1)
    for i in range(0,modelOutput.shape[1]):
        t = modelOutput[:length, i].tolist() # for covid19
        t.sort()
        if i==0:
            averageEnergies.append(np.mean(t[-1:]))
        else:
            averageEnergies2.append(np.mean(t[-topn:]))
    averageEnergies2=np.max(averageEnergies2)
    averageEnergies=np.array([averageEnergies[0],averageEnergies2])
    averageEnergies = averageEnergies / np.sum(averageEnergies, keepdims=True)
    pred=np.argmax(averageEnergies)
    if pred >=1:
        pred=1
    else:
        pred=0
    label=labels.cpu().numpy()
    if label>=1:
        label=1
    else:
        label=0
    iscorrect = label == pred
    return averageEnergies.tolist(), [iscorrect],pred

def _validate_cap_covid(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    output=np.exp(modelOutput.cpu().numpy())[:length, [1,3]]
    output=output/np.sum(output,1,keepdims=True)
    for i in range(output.shape[1]):
        t = output[:,i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        averageEnergies.append(np.mean(t[-topn:]))
    pred=np.argmax(averageEnergies)
    label=labels.cpu().numpy()
    if label==1:
        label=0
    else:
        label=1
    iscorrect = label == pred
    return averageEnergies, [iscorrect],pred

def _validate_hxnx_covid(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    output=np.exp(modelOutput.cpu().numpy())[:length, [2,3]]
    output = output / np.sum(output, 1, keepdims=True)
    for i in range(output.shape[1]):
        t = output[:,i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        averageEnergies.append(np.mean(t[-topn:]))
    pred=np.argmax(averageEnergies)
    label=labels.cpu().numpy()
    if label==2:
        label=0
    else:
        label=1
    iscorrect = label == pred
    return averageEnergies, [iscorrect],pred

def _validate_multicls(modelOutput, labels,length,topn=3):
    averageEnergies=[]
    for i in range(0,modelOutput.shape[1]):
        t = np.exp(modelOutput.cpu().numpy())[:length, i].tolist() # for covid19
        #pos_count = np.sum(np.array(modelOutput) > 0.5)
        t.sort()
        if i==0:
            averageEnergies.append(np.mean(t[-topn:]))#
        else:
            averageEnergies.append(np.mean(t[-topn:]))
    pred=np.argmax(averageEnergies)
    iscorrect = labels.cpu().numpy() == pred

    return averageEnergies, iscorrect,pred


class Validator():
    def __init__(self, options, mode,model,savenpy=None):
        self.R = 'R' in options['general'].keys()
        self.cls_num=options['general']['class_num']
        self.use_plus=options['general']['use_plus']
        self.use_3d = options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice = options['general']['use_slice']
        self.asinput = options['general']['plus_as_input']
        mod=options['general']['mod']
        #datalist = args.imgpath
        #masklist =args.maskpath
        self.savenpy = savenpy
        if mod=='healthy':
            f='data/lists/reader_healthy_vs_ill.list'
        elif mod=='cap':
            f = 'data/lists/reader_cap_vs_covid.list'
        elif mod=='AB-in':
            f = 'data/lists/reader_influenza_vs_covid.list'
        elif mod=='ind':
            f = 'data/lists/ind_list2.list'
        elif mod=='xct':
            f = 'data/lists/testlist_xct.list'
        else:
            f = 'data/trainlist_ct_only.list'

        self.model=model
        self.mod=mod

        if mod=='ind':
            self.validationdataset = IndtestDataset(options[mode]["data_root"],
                                                    options[mode]["padding"],
                                                    f,cls_num=self.cls_num,mod=options['general']['mod'],
                                                    options=options)
        else:
            self.validationdataset = NCPJPGtestDataset_new(options[mode]["data_root"],
                                                           options[mode]["padding"],
                                                            f,cls_num=self.cls_num,mod=options['general']['mod'],
                                                           options=options)

        self.topk=3
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
            self.validationdataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            drop_last=False
        )
        self.mode = mode
        self.epoch = 0

    def __call__(self):
        self.epoch += 1
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((self.cls_num + self.use_plus * 2*(1-self.asinput)))
            Matrix = np.zeros((self.cls_num, self.cls_num))
            if self.cls_num>2:
                if self.mod=='healthy':
                    validator_function=_validate_healthy_or_not#win0
                elif self.mod== 'cap':
                    validator_function = _validate_cap_covid
                elif self.mod== 'AB-in':
                    validator_function = _validate_hxnx_covid
                elif self.mod=='ind':
                    validator_function = _validate_multicls
                elif self.mod=='xct':
                    validator_function = _validate_cap_covid
                else:
                    validator_function = _validate_multicls
            else:
                validator_function = _validate_cp
            self.model.eval()
            LL = []
            GG=[]
            AA=[]
            if (self.usecudnn):
                net = nn.DataParallel(self.model).cuda()
            num_samples = np.zeros((self.cls_num + self.use_plus * 2*(1-self.asinput)))
            tic=time.time()
            X=[]
            Y=[]
            Z=[]
            P=[]
            N=[]
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda().float()
                labels = Variable(sample_batched['label']).cuda()

                if self.use_plus:
                    age = Variable(sample_batched['age']).cuda()
                    gender = Variable(sample_batched['gender']).cuda()
                    pos=Variable(sample_batched['pos']).cuda()
                name =sample_batched['length'][0]
                valid_length=sample_batched['length'][1]

                rs=input.shape
                input=input.squeeze(0)
                input=input.permute(1,0,2,3)
                if input.shape[0]<3:
                    print('E')
                    continue
                if not self.use_plus:
                    try:
                        outputs,deep_feaures = net(input.float(),False)
                    except:
                        #print(input.shape)
                        continue
                else:
                    if self.asinput:
                        outputs, _, _, _, deep_feaures = net(input,pos,gender,age)
                    else:
                        outputs, out_gender, out_age,out_pos,deep_feaures = net(input)
                if SAVE_DEEP:
                    deep_feaures=deep_feaures.cpu().numpy()
                    I_r=input.cpu().numpy()[:]
                    X.append(deep_feaures)
                    Z.append(name)
                    Y.append(labels.cpu().numpy()[0][0])
                if KEEP_ALL:
                    all_numpy=np.exp(outputs.cpu().numpy()[:valid_length,1])
                    np.save('multi_period_scores/npys_re/'+name[0].split('/')[-1]+'.npy',all_numpy)

                (vector, isacc,pos_count) = validator_function(outputs, labels,valid_length,self.topk)
                _, maxindices = outputs.cpu().max(1)

                output_numpy = vector
                label_numpy = labels.cpu().numpy()[0, 0]
                if isacc[0]==False:
                    print(name[0],isacc,vector,input.shape[0])
                if self.mod=='healthy':
                    if label_numpy>=1:
                        label_numpy=1
                    else:
                        label_numpy=0
                elif self.mod=='cap' or self.mod=='xct':
                    if label_numpy==1:
                        label_numpy=0
                    else:
                        label_numpy=1
                elif self.mod=='AB-in':
                    if label_numpy==2:
                        label_numpy=0
                    else:
                        label_numpy=1

                # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    LL.append([name[0]]+ output_numpy+[label_numpy])
                    Matrix[label_numpy, pos_count] += 1
                    #if isacc[i]==0:
                    #(name[0]+'\t'+str(all_numpy)+'\t'+str(pos_count)+'\t'+str(np.array(slice_idx).tolist())+'\n')
                    if isacc[i] == 1:
                        count[label_numpy] += 1
                    num_samples[label_numpy] += 1
                    if i_batch%100==0 and i_batch>1:
                        #print(count[:self.cls_num].sum() / num_samples[:self.cls_num].sum(), np.mean(AA))
                        print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),
                                                                                      count.sum(),len(self.validationdataset),
                                                                                       1.0*count/num_samples))
                if True and self.mod=='all':
                    if labels[0] == 3:
                        prob = torch.exp(outputs)[:,-1].detach().cpu().numpy()
                        #prob_idx=np.argsort(prob)
                        for idd in range(outputs.shape[0]):
                            I = np.array(input[idd, :, :, :].cpu().numpy() * 255, np.uint8).transpose(1, 2, 0) [:,:, [2, 1, 0]]
                            J = I[I[:, :, 2] == 255, 1].mean()
                            if  J>50.5:
                                cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train2/abnor/abnor_' +
                                            name[0].split('/')[-1].split('.')[0]+'_'+str(idd)+'.jpg',I)
                            if J<27.5:
                                cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train2/nor/nor_' +
                                            name[0].split('/')[-1].split('.')[0]+'_'+str(idd)+'.jpg',I)



        print(count[:self.cls_num].sum() / num_samples[:self.cls_num].sum(),np.mean(AA))
        LL = np.array(LL)
        print(Matrix)
        np.save(self.savenpy, LL)
        if SAVE_DEEP:
            X=np.array(X)
            Y=np.array(Y)
            Z = np.array(Z)
            np.save(os.path.join('saves','X.npy'),X)
            np.save(os.path.join('saves', 'Y.npy'), Y)
            np.save(os.path.join('saves', 'Z.npy'), Z)
        if self.use_plus and not self.asinput:
            GG = np.array(GG)
            AA=np.array(AA)
            np.save('gender.npy', GG)
            np.save('age.npy', AA)
        toc=time.time()
        print((toc-tic)/self.validationdataloader.dataset.__len__())
        return count / num_samples, count[:self.cls_num].sum() / num_samples[:self.cls_num].sum()

    def age_function(self, pre, label):
        pre=pre.cpu().numpy().mean()* 90
        label=label.cpu().numpy()
        return np.mean(pre-label),pre


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--deepsave", help="A path to save deepfeature", type=str,
                        # default='re/cap_vs_covid.npy')
                        default='deep_f')
    parser.add_argument("-e", "--exclude_list",
                        help="A path to a txt file for excluded data list. If no file need to be excluded, "
                             "it should be 'none'.", type=str,
                        default='none')
    parser.add_argument("-v", "--invert_exclude", help="Whether to invert exclude to include", type=bool,
                        default=False)
    parser.add_argument("-g", "--gpuid", help="gpuid", type=str,
                        default='4')
    args = parser.parse_args()
    os.makedirs(args.deepsave, exist_ok=True)

    print("Loading options...")
    with open('test.toml', 'r') as optionsFile:
        options = toml.loads(optionsFile.read())

    if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']

    torch.manual_seed(options["general"]['random_seed'])

    # Create the model.
    if options['general']['use_plus']:
        model = resnet152_plus(options['general']['class_num'])
    else:
        model = resnet152(options['general']['class_num'])
    if 'R' in options['general'].keys():
        model = resnet152_R(options['general']['class_num'])
    pretrained_dict = torch.load(options['general']['pretrainedmodelpath'])
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    print('matched keys:', len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    tester = Validator(options, 'test',model,options['validation']['saves'])

    result, re_all = tester()
    print (tester.savenpy)
    print('-' * 21)
    print('All acc:' + str(re_all))
    print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
    for i in range(result.shape[0]):
        print('{:<10}|{:>10}'.format(i, result[i]))
    print('-' * 21)

if __name__ == "__main__":
    main()
