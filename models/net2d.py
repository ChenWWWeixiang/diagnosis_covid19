from __future__ import print_function, division, absolute_import
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import types
import torch
import re
import torch.nn as nn

#################################################################
# You can find the definitions of those models here:
# https://github.com/pytorch/vision/blob/master/torchvision/models
#
# To fit the API, we usually added/redefined some methods and
# renamed some attributs (see below for each models).
#
# However, you usually do not need to see the original model
# definition from torchvision. Just use `print(model)` to see
# the modules and see bellow the `model.features` and
# `model.classifier` definitions.
#################################################################

__all__ = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inceptionv3',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'
]

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth',
    'densenet169': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth',
    'densenet201': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth',
    'densenet161': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth',
    'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # 'vgg16_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth',
    # 'vgg19_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth'
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 2
        }
    }

# for model_name in ['vgg16', 'vgg19']:
#     pretrained_settings[model_name]['imagenet_caffe'] = {
#         'url': model_urls[model_name + '_caffe'],
#         'input_space': 'BGR',
#         'input_size': input_sizes[model_name],
#         'input_range': [0, 255],
#         'mean': [103.939, 116.779, 123.68],
#         'std': [1., 1., 1.],
#         'num_classes': 1000
#     }

def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def load_pretrained(model, num_classes, settings):
    #assert num_classes == settings['num_classes'], \
    #    "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict,strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model




        # return loss/transposed.size(0)
#################################################################
# AlexNet

def modify_alexnet(model):
    # Modify attributs
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.linear0 = model.classifier[1]
    model.relu0 = model.classifier[2]
    model.dropout1 = model.classifier[3]
    model.linear1 = model.classifier[4]
    model.relu1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier


    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x=F.dropout(x,0.5)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def alexnet(num_classes=1000, pretrained='imagenet'):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    model = models.alexnet(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['alexnet'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_alexnet(model)
    return model

###############################################################
# DenseNets

def modify_densenets(model,num_of_cls):
    # Modify attributs
    num_of_features=2208
    #hidden=256
    model.classifier = torch.nn.Linear(num_of_features,num_of_cls)
    #model.lstm=torch.nn.LSTM(num_of_features,hidden,bidirectional=True,batch_first=True)
    #model.lstm2 = torch.nn.LSTM(hidden*2, 256, bidirectional=True,batch_first=True)
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)#padding,-1
        #x,_ = torch.max(x,0,keepdim=True)
        #x = F.dropout(x,0.5)
        x = self.classifier(x)
        x = F.log_softmax(x,-1)
        return x
    def lstm_logits(self,features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(1,x.size(0), -1)#1,padding,-1
        x,_ = self.lstm(x)
        x,_ = self.lstm2(x)
        x = F.dropout(x, 0.5)
        x = self.last_linear(x).log_softmax(-1)
        return x

    def forward(self, input):
        #torch.squeeze()
        #input=input.squeeze(0).permute(1,0,2,3)
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    #model.lstm_logits = types.MethodType(lstm_logits, model)
    return model

def densenet121(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet121'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

def densenet169(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet169'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

def densenet201(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet201'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

def densenet161(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet161'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model,num_classes)
    return model

###############################################################
# InceptionV3

def inceptionv3(num_classes=1000, pretrained='imagenet'):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    """
    model = models.inception_v3(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['inceptionv3'][pretrained]
        model = load_pretrained(model, num_classes, settings)

    # Modify attributs
    model.last_linear = model.fc
    del model.fc

    def features(self, input):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(input) # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x) # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x) # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x) # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x) # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2) # 35 x 35 x 192
        x = self.Mixed_5b(x) # 35 x 35 x 256
        x = self.Mixed_5c(x) # 35 x 35 x 288
        x = self.Mixed_5d(x) # 35 x 35 x 288
        x = self.Mixed_6a(x) # 17 x 17 x 768
        x = self.Mixed_6b(x) # 17 x 17 x 768
        x = self.Mixed_6c(x) # 17 x 17 x 768
        x = self.Mixed_6d(x) # 17 x 17 x 768
        x = self.Mixed_6e(x) # 17 x 17 x 768
        if self.training and self.aux_logits:
            self._out_aux = self.AuxLogits(x) # 17 x 17 x 768
        x = self.Mixed_7a(x) # 8 x 8 x 1280
        x = self.Mixed_7b(x) # 8 x 8 x 2048
        x = self.Mixed_7c(x) # 8 x 8 x 2048
        return x

    def logits(self, features):
        x = F.avg_pool2d(features, kernel_size=8) # 1 x 1 x 2048
        x = F.dropout(x, training=self.training) # 1 x 1 x 2048
        x = x.view(x.size(0), -1) # 2048
        x = self.last_linear(x) # 1000 (num_classes)
        if self.training and self.aux_logits:
            aux = self._out_aux
            self._out_aux = None
            return x, aux
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

###############################################################
# ResNets

def modify_resnets_r(model,num_of_cls):
    # Modify attributs
    model.bn=torch.nn.BatchNorm1d(479)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(2048+479,num_of_cls))
                                           #torch.nn.ReLU(),
                                          # torch.nn.Dropout(),
                                          # torch.nn.Linear(1024,num_of_cls))
    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,r,test=False):
        x = self.features_func(input)
        x = x.view(x.size(0), -1)
        r=r.squeeze(1)
        r=self.bn(r)
        x=torch.cat([x,r],-1)
        if test:
            x = x.max(0).values
            x=x.unsqueeze(0)
        x = self.classifier(x)
        return x.log_softmax(-1)

    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resnets(model,num_of_cls,USE_25D):
    # Modify attributs
    model.classifier = torch.nn.Linear(2048,num_of_cls)
    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,test=False):
        x = self.features_func(input)
        x = x.view(x.size(0), -1)
        if test:
            x=x.max(0).values
            x = x.unsqueeze(0)
        d = x.max(0).values
        d = d.unsqueeze(0)
        x = self.classifier(x).log_softmax(-1)
        return x,d

    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resnets_plus2(model,num_of_cls,asinput=False,USE_25D=False):
    # Modify attributs
    if USE_25D:
        model.classifier = torch.nn.Linear(2048 + 7, num_of_cls)
    else:
        model.fusing = torch.nn.Linear(4096, 1024)
        model.classifier = torch.nn.Linear(1024+6,num_of_cls)
        model.droplayer=torch.nn.Dropout(0.5)
    if not asinput:
        model.classifier_gender = torch.nn.Linear(2048, 2)
        model.classifier_age = torch.nn.Linear(1024, 5)
        model.regress_pos = torch.nn.Linear(1024, 1)


    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,ipos=None,igender=None,iage=None):
        x = self.features_func(input)
        x = x.view(x.size(0), -1)

        gender = self.classifier_gender(x).log_softmax(-1)
        x=torch.cat([gender[:,0:1].exp()*x,gender[:,1:2].exp()*x],-1)
        cc=self.droplayer(self.fusing(x).relu())
        age = self.classifier_age(cc).log_softmax(-1)
        pos = self.regress_pos(cc).sigmoid()
        f=torch.cat([age.exp(),pos, cc], -1)
        y = self.classifier(f).log_softmax(-1)
        return y, gender, age, pos, f
    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resnets_plus(model,num_of_cls,asinput=False,USE_25D=False):
    # Modify attributs
    if USE_25D:
        model.classifier = torch.nn.Linear(2048 + 7, num_of_cls)
    else:
        model.classifier = torch.nn.Linear(2048+8,num_of_cls)
    if not asinput:
        model.classifier_gender = torch.nn.Linear(2048, 2)
        model.classifier_age = torch.nn.Linear(2048, 5)
        model.regress_pos = torch.nn.Linear(2048, 1)

    model.features=torch.nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,
                                       model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool)
    del model.fc

    def features_func(self, input):
        x = self.features(input)
        return x

    def forward(self, input,ipos=None,igender=None,iage=None):
        x = self.features_func(input)
        x = x.view(x.size(0), -1)
        if USE_25D:
            x = x.max(0)
        if not asinput:
            pos=self.regress_pos(x).sigmoid()
            gender = self.classifier_gender(x)
            age = self.classifier_age(x)
        else:
            pos=ipos
            #gender=igender
            gender = torch.zeros(igender.shape[0], 2).cuda().scatter_(1, igender, 1)
            age = torch.zeros(iage.shape[0], 5).cuda().scatter_(1, iage//20, 1)
           # age=iage.float()
        if USE_25D:
            cc=torch.cat([gender.relu(),age.relu(),x],-1)
            y = self.classifier(cc).log_softmax(-1)
            return y, gender.log_softmax(-1), age.log_softmax(-1), cc, cc
        else:
            cc = torch.cat([gender.relu(), age.relu(),pos, x], -1)
            y = self.classifier(cc).log_softmax(-1)
            return y, gender.log_softmax(-1), age.log_softmax(-1), pos, cc
    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model

def modify_resUnets(model,num_of_cls):
    # Modify attributs
    model.classifier = torch.nn.Linear(2048,num_of_cls)
    model.decoder=Decoder([2048,1024,512,256,64],[1024,512,256,64])
    model.upper=nn.Sequential(nn.Upsample(scale_factor=4),DoubleConv(64,1,True))
    del model.fc

    def features_func(model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        f0 = model.maxpool(x)

        f1 = model.layer1(f0)
        f2 = model.layer2(f1)
        f3 = model.layer3(f2)
        f4 = model.layer4(f3)
        return f4,f3,f2,f1,f0

    def forward(self, input):
        Fs = self.features_func(input)
        y=self.decoder(Fs)
        x = self.avgpool(Fs[0])
        x = x.view(x.size(0), -1)
        x = self.classifier(x).log_softmax(-1)
        y = self.upper(y)
        return x,y
    # Modify methods
    model.features_func = types.MethodType(features_func, model)
    #model.classifier = types.MethodType(classifier, model)
    model.forward = types.MethodType(forward, model)
    return model
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,sigmoid=False):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if sigmoid:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Decoder(torch.nn.Module):
    def __init__(self,inner_fs,o_fs):
        super(Decoder,self).__init__()
        self.deconv1=Up(inner_fs[0]+inner_fs[1],o_fs[0])
        self.deconv2=Up(inner_fs[2]+o_fs[0],o_fs[1])
        self.deconv3 = Up(inner_fs[3]+o_fs[1], o_fs[2])
        self.deconv4 = Up(inner_fs[4]+o_fs[2], o_fs[3])

    def forward(self,Fs):
        x=self.deconv1(Fs[0],Fs[1])
        x = self.deconv2(x, Fs[2])
        x = self.deconv3(x, Fs[3])
        x = self.deconv4(x, Fs[4])
        return x

def resnet18(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-18 model.
    """
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet18'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model

def resnet34(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-34 model.
    """
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet34'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model

def resnet50(num_classes=1000, pretrained='imagenet',USE_25D=False):
    """Constructs a ResNet-50 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model,num_classes,USE_25D=USE_25D)
    return model

def resnet101(num_classes=1000, pretrained='imagenet'):
    """Constructs a ResNet-101 model.
    """
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet101'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model,num_classes)
    return model

def resnet152(num_classes=1000, pretrained='imagenet',USE_25D=False):
    """Constructs a ResNet-152 model.
    """
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['resnet152'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model,num_classes,USE_25D=USE_25D)
    return model


def resUnet152(num_classes=1000, pretrained='imagenet'):
    model = models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152'][pretrained]
    model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model,num_classes)
    return model

def resnet152_plus(num_classes=1000, pretrained='imagenet',asinput=False,USE_25D=False):
    model = models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152'][pretrained]
    model = load_pretrained(model, num_classes, settings)
    model = modify_resnets_plus2(model,num_classes,asinput=asinput,USE_25D=USE_25D)
    return model
def resnet152_R(num_classes=1000, pretrained='imagenet',asinput=False,USE_25D=False):
    model = models.resnet152(pretrained=False)
    settings = pretrained_settings['resnet152'][pretrained]
    model = load_pretrained(model, num_classes, settings)
    model = modify_resnets_r(model,num_classes)
    return model

###############################################################
# SqueezeNets

def modify_squeezenets(model,numclas):
    # /!\ Beware squeezenets do not have any last_linear module

    # Modify attributs
    model.dropout = model.classifier[0]
    #model.last_conv = model.classifier[1]
    model.relu = model.classifier[2]
    model.avgpool = model.classifier[3]
    model.last_conv = torch.nn.Conv2d(512, numclas, kernel_size=1)
    del model.classifier


    def logits(self, features):
        x = self.dropout(features)
        x = self.last_conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x).squeeze(-1).squeeze(-1)
        x=F.log_softmax(x,1)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def squeezenet1_0(num_classes=1000, pretrained='imagenet'):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['squeezenet1_0'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model

def squeezenet1_1(num_classes=1000, pretrained='imagenet'):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = models.squeezenet1_1(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['squeezenet1_1'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model,num_classes)
    return model

###############################################################
# VGGs

def modify_vggs(model,num_of_cls):
    # Modify attributs
    #model._features = model.features
    #del model.features
    #model.linear0 = model.classifier[0]
    #model.relu0 = model.classifier[1]
    #model.dropout0 = model.classifier[2]
    #model.linear1 = model.classifier[3]
    #model.relu1 = model.classifier[4]
    #model.dropout1 = model.classifier[5]
    #model.last_linear =
    model.classifier[6]=torch.nn.Linear(4096,num_of_cls)

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.dropout1(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x=x.view(x.size(0), -1)
        x = self.classifier(x)
        x=F.log_softmax(x,-1)
        return x

    # Modify methods
    #model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model

def vgg11(num_classes=1000, pretrained='imagenet'):
    """VGG 11-layer model (configuration "A")
    """
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg11'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg11_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 11-layer model (configuration "A") with batch normalization
    """
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg11_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg13(num_classes=1000, pretrained='imagenet'):
    """VGG 13-layer model (configuration "B")
    """
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg13'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg13_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 13-layer model (configuration "B") with batch normalization
    """
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg13_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg16(num_classes=1000, pretrained='imagenet'):
    """VGG 16-layer model (configuration "D")
    """
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg16'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg16_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 16-layer model (configuration "D") with batch normalization
    """
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg16_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model

def vgg19(num_classes=1000, pretrained='imagenet'):
    """VGG 19-layer model (configuration "E")
    """
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg19'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model,num_classes)
    return model

def vgg19_bn(num_classes=1000, pretrained='imagenet'):
    """VGG 19-layer model (configuration 'E') with batch normalization
    """
    model = models.vgg19_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['vgg19_bn'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model,num_classes)
    return model