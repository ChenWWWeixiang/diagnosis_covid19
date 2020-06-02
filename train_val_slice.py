from __future__ import print_function
from models.Dense3D import Dense3D
import torch
import toml,glob
from training import Trainer
from testengine import Validator
from validation import Validator as Validator2
import torch.nn as nn
import os
from models.net2d import densenet161, resnet152, resnet152_plus, resnet152_R
import warnings
def renew_list():
    path=[
        '/mnt/data9/covid_detector_jpgs/selected_train2/nor',
        '/mnt/data9/covid_detector_jpgs/selected_train2/abnor',
        '/mnt/data9/covid_detector_jpgs/selected_train1/nor',
         '/mnt/data9/covid_detector_jpgs/selected_train1/abnor',
         ]
    f1 = open('data/txt/train_slice.txt', 'w')
    for ipath in path:
        files = os.listdir(ipath)
        if len(files) == 0:
            continue
        names_id = [file.split('_')[0] + '_' + file.split('_')[1] + '_' + file.split('_')[2] for file in files]
        names_id = list(set(names_id))
        train = names_id
        for _, i in enumerate(train):
            names = glob.glob(ipath + '/' + i + '_*')
            for name in names:
                f1.writelines(name + '\n')
warnings.filterwarnings("ignore")

print("Loading options...")
with open('options_slice.toml', 'r') as optionsFile:
    # with open('options_lip.toml', 'r') as optionsFile:
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
        model = resnet152_plus(options['general']['class_num'], asinput=options['general']['plus_as_input'],
                               USE_25D=options['general']['use25d'])
    else:
        model = resnet152(options['general']['class_num'],
                          USE_25D=options['general']['use25d'])  # vgg19_bn(2)#squeezenet1_1(2)
    if 'R' in options['general'].keys():
        model = resnet152_R(options['general']['class_num'])
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
    if options['general']['mod'] == 'slice':
        validator = Validator2(options, 'validation', model, savenpy=options["validation"]["saves"],
                               )
    else:
        validator = Validator(options, 'validation', model, savenpy=options["validation"]["saves"],
                              )  # TODO:change mod
if (options['test']['test']):
    if options['general']['mod'] == 'slice':
        tester = Validator2(options, 'test', model, savenpy=options["test"]["saves"],
                               )
    else:
        tester = Validator(options, 'test', model, savenpy=options["test"]["saves"],
                              )  # TODO:change mod

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):
    if (options["training"]["train"]):
        trainer(epoch)
    if (options["validation"]["validate"]) and epoch % 5 == 0:
        result, re_all = validator()
        # trainer.ScheduleLR(result.min())
        print(options['training']['save_prefix'])
        print('-' * 21)
        print('All acc:' + str(re_all))
        print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
        for i in range(len(result)):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-' * 21)
        renew_list()

if (options['test']['test']):
    result, re_all = tester()
    print('-' * 21)
    print('All acc:' + str(re_all))
    print('{:<10}|{:>10}'.format('Cls #', 'Accuracy'))
    for i in range(2):
        print('{:<10}|{:>10}'.format(i, result[i]))
    print('-' * 21)

Trainer.writer.close()
print(options['training']['save_prefix'])