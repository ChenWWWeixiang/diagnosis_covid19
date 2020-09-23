from torch.autograd import Variable
import warnings

warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data.dataset import NCPDataset, NCP2DDataset, NCPJPGDataset, NCPJPGDataset_new, NCPJPGtestDataset_new
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import os, cv2, shutil
import pdb
import math

USE_25D = False


class NLLSequenceLoss(torch.nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """

    def __init__(self, w=[0.55, 0.45]):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = torch.nn.NLLLoss(reduction='none', weight=torch.Tensor([0.8, 0.8, 0.8, 0.5]).cuda())

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i,], target).unsqueeze(1))
        loss = torch.cat(loss, 1)
        # print('loss:',loss)
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()

        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, :L - 1] = 1.0
        # print('mask:',mask)
        # print('mask * loss',mask*loss)
        loss = (loss * mask).sum() / mask.sum()
        return loss


def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)


def output_iteration(loss, i, time, totalitems):
    avgBatchTime = time / (i + 1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time),
                                                                                           timedelta_string(estTime),
                                                                                           loss))


class Trainer():
    tot_iter = 0
    writer = SummaryWriter()

    def __init__(self, options, model):
        self.R = 'R' in options['general'].keys()
        self.cls_num = options['general']['class_num']
        self.use_plus = options['general']['use_plus']
        self.use_slice = options['general']['use_slice']
        self.usecudnn = options["general"]["usecudnn"]
        self.use_3d = options['general']['use_3d']
        self.batchsize = options["input"]["batchsize"]
        self.use_lstm = options["general"]["use_lstm"]
        self.statsfrequency = options["training"]["statsfrequency"]
        self.learningrate = options["training"]["learningrate"]
        self.modelType = options["training"]["learningrate"]
        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        self.save_prefix = options["training"]["save_prefix"]
        self.asinput = options['general']['plus_as_input']
        self.USE_25D = options['general']['use25d']
        if options['general']['use_slice']:
            if self.USE_25D:
                f = 'data/3cls_train.list'
                self.trainingdataset = NCPJPGtestDataset_new(options["training"]["data_root"],
                                                             options["training"]["padding"],
                                                             f, cls_num=self.cls_num, mod=options['general']['mod'],
                                                             options=options)
            else:
                self.trainingdataset = NCPJPGDataset_new(options["training"]["data_root"],
                                                         options["training"]["index_root"],
                                                         options["training"]["padding"],
                                                         True, cls_num=self.cls_num, mod=options['general']['mod'],
                                                         options=options)
        else:
            if options['general']['use_3d']:
                self.trainingdataset = NCPDataset(
                    options["training"]["index_root"],
                    options["training"]["padding"],
                    True,
                    z_length=options["model"]["z_length"])
            else:
                self.trainingdataset = NCP2DDataset(options["training"]["data_root"],
                                                    options["training"]["index_root"],
                                                    options["training"]["padding"],
                                                    True)  ##TODO:3
        weights = self.trainingdataset.make_weights_for_balanced_classes()
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(self.trainingdataset))

        self.trainingdataloader = DataLoader(
            self.trainingdataset,
            batch_size=options["input"]["batchsize"],
            # shuffle=options["input"]["shuffle"],
            num_workers=options["input"]["numworkers"],
            drop_last=False,
            sampler=sampler
        )

        self.optimizer = optim.Adam(model.parameters(), lr=self.learningrate, amsgrad=True)
        # self.schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'max',
        #                                                         patience=3, factor=.3, threshold=1e-3, verbose=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.model = model
        if self.use_3d:
            self.criterion = self.model.loss()
        else:
            # criterion=nn.
            # w=torch.Tensor(self.trainingdataset.get_w()).cuda()
            # print(w)
            # w = torch.Tensor(w).cuda()
            self.criterion = nn.NLLLoss().cuda()  # 0.3,0.7
            if self.use_plus:
                self.criterion_age = nn.NLLLoss(ignore_index=-1).cuda()
                self.criterion_gender = nn.NLLLoss(ignore_index=-1,
                                                   weight=torch.Tensor([0.3, 0.7]).cuda()).cuda()
                self.criterion_pos = nn.SmoothL1Loss().cuda()
        if self.use_lstm:
            self.criterion = NLLSequenceLoss()
        if (self.usecudnn):
            self.net = nn.DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()

    def ScheduleLR(self, acc):
        self.scheduler.step(acc)

    def __call__(self, epoch):
        # set up the loss function.
        self.model.train()
        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.trainingdataloader):
            self.optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            length = len(sample_batched['length'][1])
            features = Variable(sample_batched['features'])
            name = sample_batched['name']
            if self.use_plus:
                age = Variable(sample_batched['age']).cuda()
                gender = Variable(sample_batched['gender']).cuda()
                pos = Variable(sample_batched['pos']).cuda()
            # break
            if self.USE_25D:
                input = input.squeeze(0)
                input = input.permute(1, 0, 2, 3)
                features = features.squeeze(0)
            if (self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()
            if not self.use_plus:
                if self.R:
                    outputs, _ = self.net(input, features, True)
                else:
                    outputs, _ = self.net(input, False)
            else:
                if self.asinput:
                    outputs, _, _, _, deep_feaures = self.net(input, pos, gender, age)
                else:
                    outputs, out_gender, out_age, out_pos, deep_feaures = self.net(input)
            if self.use_3d or self.use_lstm:
                loss = self.criterion(outputs, length, labels.squeeze(1))
            else:
                loss = self.criterion(outputs, labels.squeeze(1))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            sampleNumber = i_batch * self.batchsize

            if (sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime,
                                 len(self.trainingdataset))
                Trainer.writer.add_scalar('Train Loss', loss, Trainer.tot_iter)
            Trainer.tot_iter += 1

        print("Epoch " + str(epoch) + "completed, saving state...")
        print(self.use_3d)
        torch.save(self.model.state_dict(), "{}.pt".format(self.save_prefix))
