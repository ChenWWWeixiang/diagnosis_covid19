from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from datetime import datetime, timedelta

from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import NCPDataset,NCP2DDataset,NCPJPGDataset,NCPJPGDataset_new,NCPJPGtestDataset_new
import os,cv2
import numpy as np

def _validate(modelOutput, length, labels, total=None, wrong=None):

    averageEnergies = torch.mean(modelOutput.data, 1)
    for i in range(modelOutput.size(0)):
        #print(modelOutput[i,:length[i]].sum(0).shape)
        averageEnergies[i] = modelOutput[i,:length[i]].mean(0)

    maxvalues, maxindices = torch.max(averageEnergies, 1)
    #print(maxindices.cpu().numpy())
    #print(labels.cpu().numpy())
    count = 0

    for i in range(0, labels.squeeze(1).size(0)):
        l = int(labels.squeeze(1)[i].cpu())
        if total is not None:
            if l not in total:
                total[l] = 1
            else:
                total[l] += 1
        if maxindices[i] == labels.squeeze(1)[i]:
            count += 1
        else:
            if wrong is not None:
               if l not in wrong:
                   wrong[l] = 1
               else:
                   wrong[l] += 1

    return (averageEnergies, count)

class Validator():
    def __init__(self, options, mode,model,savenpy):
        self.R='R' in options['general'].keys()
        self.model=model
        self.cls_num=options['general']['class_num']
        self.use_plus = options['general']['use_plus']
        self.use_3d=options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice=options['general']['use_slice']
        self.asinput = options['general']['plus_as_input']
        self.USE_25D = options['general']['use25d']
        if options['general']['use_slice']:
            if self.USE_25D:
                f = 'data/3cls_test.list'
                self.validationdataset = NCPJPGtestDataset_new(options[mode]["data_root"],options["training"]["padding"],
                                                             f, cls_num=self.cls_num, mod=options['general']['mod'],
                                                               options=options)
            else:
                self.validationdataset = NCPJPGDataset_new(options[mode]["data_root"],
                                                        options[mode]["index_root"],
                                                        options[mode]["padding"],
                                                        False,cls_num=self.cls_num,
                                                        mod=options['general']['mod'],
                                                           options=options)
        else:
            if options['general']['use_3d']:
                self.validationdataset = NCPDataset(  options[mode]["index_root"],
                                                      options[mode]["padding"],
                                                      False,
                                                    z_length=options["model"]["z_length"])
            else:
                self.validationdataset = NCP2DDataset(options[mode]["data_root"],
                                                        options[mode]["index_root"],
                                                        options[mode]["padding"],
                                                        False)
        self.savingnpy=savenpy

        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=True,
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode
        self.epoch=0
        
    def __call__(self):
        self.epoch+=1
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((self.cls_num+self.use_plus*2*(1-self.asinput)))
            Matrix=np.zeros((self.cls_num,self.cls_num))
            if self.use_3d:
                validator_function = self.model.validator_function()
            if self.use_lstm:
                validator_function = _validate
                self.model.eval()
            LL=[]
            GG=[]
            AA=[]
            if(self.usecudnn):
                net = nn.DataParallel(self.model).cuda()
            error_dir='error/'
            os.makedirs(error_dir,exist_ok=True)
            cnt=0
            num_samples = np.zeros((self.cls_num+self.use_plus*2*(1-self.asinput)))
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                #length = len(sample_batched['length'][1])
                names=sample_batched['name']
                outputs,f = net(input)

                output_numpy = np.exp(outputs.cpu().numpy())
                output_numpy=output_numpy[:,[0,-1]]
                #output_numpy_ab=np.sum(output_numpy[:,1:],1)
                #output_numpy=np.stack([output_numpy[:,0],output_numpy_ab],-1)
                output_numpy=output_numpy/output_numpy.sum(1,keepdims=True)
                pre=np.array(output_numpy[:,-1]>0.5,np.int)
                isacc=labels.cpu().numpy().reshape(labels.size(0))==pre
                label_numpy=labels.cpu().numpy()[:,0]

               # argmax = (-vector.cpu().numpy()).argsort()
                for i in range(labels.size(0)):
                    LL.append([names[i], output_numpy[i,-1], label_numpy[i]])
                    Matrix[label_numpy[i],pre[i]]+=1
                    if isacc[i]==1 :
                        count[labels[i]] += 1
                    num_samples[labels[i]]+=1
                    if self.mode=='validation':
                        if labels[i] == 1 and output_numpy[i,-1]>0.99:
                            I = np.array(input[i, :, :, :].cpu().numpy() * 255, np.uint8).transpose(1, 2, 0) [:,:, [2, 1, 0]]
                            cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train1/abnor/abnor_' +
                                        names[i].split('/')[-1].split('.')[0]+'.jpg',I)
                        if labels[i] == 0 and output_numpy[i,0]>0.99:
                            I = np.array(input[i, :, :, :].cpu().numpy() * 255, np.uint8).transpose(1, 2, 0) [:,:, [2, 1, 0]]
                            cv2.imwrite('/mnt/data9/covid_detector_jpgs/selected_train1/nor/nor_' +
                                        names[i].split('/')[-1].split('.')[0]+'.jpg',I)


        print(count[:self.cls_num].sum()/num_samples[:self.cls_num].sum(),np.mean(AA))
        LL=np.array(LL)
        np.save(self.savingnpy, LL)

        print(Matrix)
        return count/num_samples,count[:self.cls_num].sum()/num_samples[:self.cls_num].sum()