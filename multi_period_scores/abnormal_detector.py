from __future__ import print_function
from models.Dense3D import Dense3D
import torch
import toml
from training import Trainer
from validation import Validator
import torch.nn as nn
import os
from models.net2d import densenet121, densenet161, squeezenet1_1, vgg19_bn, resnet152, resnet152_plus

print("Loading options...")
with open('get_abnormal_slices.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if (options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']

torch.manual_seed(options["general"]['random_seed'])

# Create the model.
if options['general']['use_3d']:
    model = Dense3D(options)  ##TODO:1
elif options['general']['use_slice']:
    if options['general']['use_plus']:
        model = resnet152_plus(options['general']['class_num'])
    else:
        model = resnet152(options['general']['class_num'])  # vgg19_bn(2)#squeezenet1_1(2)
else:
    model = densenet161(2)

if (options["general"]["loadpretrainedmodel"]):
    # remove paralle module
    pretrained_dict = torch.load(options["general"]["pretrainedmodelpath"])
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    print('matched keys:', len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Move the model to the GPU.
# criterion = model.loss()

if (options["general"]["usecudnn"]):
    torch.cuda.manual_seed(options["general"]['random_seed'])
    torch.cuda.manual_seed_all(options["general"]['random_seed'])

if (options["training"]["train"]):
    trainer = Trainer(options, model)
if (options["validation"]["validate"]):
    validator = Validator(options, 'validation', model)
if (options['test']['test']):
    tester = Validator(options, 'test')

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):
    if (options["training"]["train"]):
        trainer(epoch)
    if (options["validation"]["validate"]):
        result, re_all = validator()
        #trainer.ScheduleLR(result[:4].min())
        print(options['training']['save_prefix'])
        print('-' * 21)
        print('All acc:' + str(re_all))
        print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-' * 21)

    if (options['test']['test']):
        result, re_all = tester(model)
        print('-' * 21)
        print('All acc:' + str(re_all))
        print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(2):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-' * 21)

Trainer.writer.close()
print(options['training']['save_prefix'])