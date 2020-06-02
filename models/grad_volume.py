import argparse
import cv2, os
import numpy as np
import torch
import SimpleITK as sitk
from torch.autograd import Function
from torchvision import models
from models.net2d import vgg19_bn, densenet161, vgg16, vgg19, resnet152

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img):
    means = [0, 0, 0]
    stds = [1, 1, 1]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        pred = np.exp(output.log_softmax(-1).cpu().data.numpy()[:, index])
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, pred


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output,f = self.forward(input.cuda())
        else:
            output,f = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--list', type=str, default='../data/lists/test.list',
                        help='path list')
    parser.add_argument('--model_path', type=str, default='../key_result/model/new_4cls_pure.pt',
                        help='Model path')
    parser.add_argument('--output_path', type=str, default='/mnt/data9/cam/jpgs',
                        help='Cam output path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

o_img_nii='/mnt/data9/cam/pre'
o_msk_nii = '/mnt/data9/cam/mask'
o_lung_nii='/mnt/data9/cam/lung'
os.makedirs(o_img_nii, exist_ok=True)
os.makedirs(o_msk_nii, exist_ok=True)
os.makedirs(o_lung_nii, exist_ok=True)
def deprocess_image(img, mask=None):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img[mask == 0] = 0.5
    img = np.clip(img, 0, 1)

    return np.uint8(img * 255)


def show_cam_on_image(img, mask, extral=None):
    if isinstance(extral, np.ndarray):
        mask = mask * extral[:, :, 1]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam[extral == 0] = np.float32(img)[extral == 0]
    # cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    return np.uint8(255 * cam)


def model_get(path):
    model = resnet152(4)
    pretrained_dict = torch.load(path)
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    # print('matched keys:', len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

import glob,random
if __name__ == '__main__':
    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    grad_cam = GradCam(model=model_get(args.model_path), \
                       target_layer_names=["7"], use_cuda=args.use_cuda)
    gb_model = GuidedBackpropReLUModel(model=model_get(args.model_path), use_cuda=args.use_cuda)
    o_path = args.output_path
    input_list = open(args.list,'r').readlines()
    random.shuffle(input_list)
    input_list=input_list[:500]
    os.makedirs(o_path, exist_ok=True)

    for names in input_list:
        img = sitk.ReadImage(names.split(',')[0])
        seg = sitk.ReadImage(names.split(',')[1][:-1])
        V=sitk.GetArrayFromImage(img)
        M = sitk.GetArrayFromImage(seg)

        #M = M[:V.shape[0], :, :]
        sums = M.sum(1).sum(1)
        idd = np.where(sums > 500)
        iddx = np.where(M > 0)
        M = M[idd[0], iddx[1].min():iddx[1].max(), iddx[2].min():iddx[2].max()]
        V = V[idd[0], iddx[1].min():iddx[1].max(), iddx[2].min():iddx[2].max()]
       # L = L[idd[0], iddx[1].min():iddx[1].max(), iddx[2].min():iddx[2].max()]
        CAM_V = []
        for idx, i in enumerate(range(0,V.shape[0],5)):
            data = V[i, :, :]
            data[data > 500] = 500
            data[data < -1200] = -1200
            data = data * 255.0 / 1700
            data = data - data.min()
            img_raw = np.stack([data,data,data], -1)
            img_raw = img_raw.astype(np.uint8)
            # data=data/data.max()
            img = np.stack([data, M[i, :, :] * data, M[i, :, :] * 255], -1)  # mask one channel
            img = img.astype(np.uint8)
            raw_shape = (img.shape[1], img.shape[0])

            img_raw = np.float32(cv2.resize(img_raw, (224, 224))) / 255
            img = np.float32(cv2.resize(img, (224, 224)))/255

            input = preprocess_image(img)
            if 'healthy'in names:
                target_index=0
                continue
            elif 'cap' in names:
                target_index=1
            elif 'HxNx' in names:
                target_index=2
            else:
                target_index=3
            #target_index=3
            mask, pred = grad_cam(input, target_index)
            cam = show_cam_on_image(img_raw, mask)
            cam = cv2.resize(cam, raw_shape)
            img_raw=cv2.resize(img_raw,raw_shape)
            I=np.concatenate([img_raw*255,cam],1)
            gb = gb_model(input, index=target_index)
            gb = gb.transpose((1, 2, 0))
            if pred>0.8:
                b = deprocess_image(gb)  ##                              gb: guided backward map
                cam_mask = cv2.merge([mask, mask, mask])
                attention_area = (cam_mask ) > 0.85
                #attention_area = attention_area * (np.abs(gb - 128) > 64)
                gbt = gb.copy()
                attention_area = np.max(attention_area,-1)
                attention_area = (attention_area >= 0.5).astype(np.uint8)
                kernel = np.ones((7, 7), np.uint8)
                attention_area = cv2.erode(cv2.erode(cv2.morphologyEx(attention_area, cv2.MORPH_CLOSE, kernel), kernel), kernel)
                attention_area=cv2.morphologyEx(attention_area, cv2.MORPH_OPEN, kernel)
                lung_mask = cv2.dilate(img[:, :, 2], kernel)
                attention_area = attention_area * lung_mask
                if np.sum(attention_area) <= 300:
                    continue
                attention_area = np.stack([attention_area, attention_area, attention_area], -1)

                # cam = show_cam_on_image(img_raw, mask)
                cam_gb = deprocess_image(cam_mask * gbt, attention_area)  ##guided gradcam
                img_raw = cv2.resize(img_raw, raw_shape)
                cam_mask = cv2.resize(cam_mask, raw_shape)
                cam_gb = cv2.resize(cam_gb, raw_shape)
                cam = cv2.resize(cam, raw_shape)
                attention_area=cv2.resize(attention_area,raw_shape)
                print(names,i,'done')
                I = np.concatenate([img_raw * 255, cam, cam_gb, attention_area * 255], 1)
                output_name=os.path.join(o_path,names.split(',')[1].split('/')[-1].split('.')[0]+'_'+str(i)+'pre'+str(pred[0])+'.jpg')
                cv2.imwrite(output_name,I)
                Inii = sitk.GetImageFromArray(img_raw[:, :, 1] * 255)
                Lnii = sitk.GetImageFromArray(img[:, :, 2])
                Mnii = sitk.GetImageFromArray(attention_area[:, :, 1])
                sitk.WriteImage(Inii, os.path.join(o_img_nii, output_name.split('/')[-1][:-4] + '.nii'))
                sitk.WriteImage(Mnii, os.path.join(o_msk_nii, output_name.split('/')[-1][:-4] + '.nii'))
                sitk.WriteImage(Lnii, os.path.join(o_lung_nii, output_name.split('/')[-1][:-4] + '.nii'))
           # CAM_V.append(cam)
