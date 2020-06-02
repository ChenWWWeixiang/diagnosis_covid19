from torch.autograd import Variable
import torch
import time
import torch.optim as optim
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import torch.nn as nn
from data.dataset import NCPJPGtestDataset_new
import os, cv2
import toml
from models.net2d import resnet152
import numpy as np
#from models.g_cam import GuidedPropo
import matplotlib as plt
KEEP_ALL=True
def _validate(modelOutput, labels, topn=1):
    modelOutput=list(np.exp(modelOutput.cpu().numpy())[:,1])
    pos_count=np.sum(np.array(modelOutput)>0.5)
    modelOutput.sort()
    averageEnergies = np.mean(modelOutput[-topn:])
    iscorrect = labels.cpu().numpy()==(averageEnergies>0.5)
    return averageEnergies,iscorrect,pos_count/len(modelOutput)
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-f", "--inputfile", help="A list of paths for lung segmentation data",  type=str,
                    default='../data/lists/test.list')
parser.add_argument("-o", "--savenpy", help="A path to save record",  type=str,
                    default='scores_mp.npy')
parser.add_argument("-p", "--modelpath", help="Whether to invert exclude to include",  type=str,
                    default='../weights/locating.pt')
parser.add_argument("-g", "--gpuid", help="gpuid",  type=str,
                    default='1')
args = parser.parse_args()

class Validator():
    def __init__(self, options, mode):
        self.use_3d = options['general']['use_3d']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_lstm = options["general"]["use_lstm"]
        self.batchsize = options["input"]["batchsize"]
        self.use_slice = options['general']['use_slice']
        datalist=args.inputfile
        self.savenpy = args.savenpy

        self.validationdataset = NCPJPGtestDataset_new(options[mode]["data_root"],
                                                       options[mode]["padding"],
                                                       datalist, cls_num=2, mod=options['general']['mod'],
                                                       options=options)

        self.topk=3
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
            self.validationdataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        self.mode = mode
        self.epoch = 0

    def __call__(self, model):
        self.epoch += 1
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((2))
            validator_function = _validate
            model.eval()
            LL = []
            if (self.usecudnn):
                net = nn.DataParallel(model).cuda()
            error_dir = 'error/'
            os.makedirs(error_dir, exist_ok=True)
            cnt = 0

            num_samples = np.zeros((2))
            tic=time.time()
            for i_batch, sample_batched in enumerate(self.validationdataloader):
                input = Variable(sample_batched['temporalvolume']).float().cuda()
                labels = Variable(sample_batched['label']).cuda()
                name =sample_batched['length'][0]
                valid_length=len(sample_batched['length'][1])

                input=input.squeeze(0)
                input=input.permute(1,0,2,3)
                outputs,_ = net(input)
                if KEEP_ALL:
                    all_numpy=np.exp(outputs.cpu().numpy()[:,1])
                    np.save('npys/'+name[0].split('/')[-1],all_numpy)
                (vector, isacc,pos_count) = validator_function(outputs, labels,self.topk)

                LL.append([name[0],pos_count])
                print([name[0],pos_count])
                #print(name[0],isacc,vector)

        print(count.sum() / num_samples.sum())
        LL = np.array(LL)
        np.save(self.savenpy, LL)
        toc=time.time()
        #print((toc-tic)/200)
        return count / num_samples, count.sum() / num_samples.sum()

print("Loading options...")
with open('get_abnormal_slices.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

torch.manual_seed(options["general"]['random_seed'])

# Create the model.
model = resnet152(2)

pretrained_dict = torch.load(args.modelpath)
# load only exists weights
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                   k in model_dict.keys() and v.size() == model_dict[k].size()}
print('matched keys:', len(pretrained_dict))
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

tester = Validator(options, 'test')

result, re_all = tester(model)
print (tester.savenpy)
print('-' * 21)
print('All acc:' + str(re_all))
print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
for i in range(2):
    print('{:<10}|{:>10}'.format(i, result[i]))
print('-' * 21)

