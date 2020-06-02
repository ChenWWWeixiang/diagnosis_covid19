import argparse
import cv2,os
import numpy as np
import torch,random
import  SimpleITK as sitk
from torch.autograd import Function
from torchvision import models
from models.net2d import vgg19_bn,densenet161,vgg16,vgg19,resnet152,resnet152_plus
os.environ['CUDA_VISIBLE_DEVICES']='3'
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

    def __init__(self, model, target_layers,use_plus=False):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)
        self.use_plus=use_plus
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        if self.use_plus:
            gender = self.model.classifier_gender(output)
            age = self.model.classifier_age(output)
            pos = self.model.regress_pos(output)
            cc = torch.cat([gender.relu(), age.relu(),output], -1)
            output = self.model.classifier(cc)
        else:
            output = self.model.classifier(output)
        return target_activations, output
def preprocess_image(img):
    means = [0,0,0]
    stds = [1,1,1]

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
    def __init__(self, model, target_layer_names, use_cuda,use_plus=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names,use_plus)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        pred = np.exp(output.log_softmax(-1).cpu().data.numpy())
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
        self.model.classifier_gender.zero_grad()
        self.model.classifier_age.zero_grad()
        self.model.regress_pos.zero_grad()
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
        return cam,pred
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
    def __init__(self, model, use_cuda,use_plus=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.use_plus=use_plus
        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)
        output=output[0]
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
    parser.add_argument('--image_path', type=str, default='/mnt/data9/covid_detector_jpgs/raw_covid',
                        help='Input raw image path')
    parser.add_argument('--mask_path', type=str, default='/mnt/data9/covid_detector_jpgs/masked_covid',
                        help='Input mask image path')
    parser.add_argument('--model_path', type=str, default='../weights/model_3cls_gender.pt',
                        help='Model path')
    parser.add_argument('--output_path', type=str, default='/mnt/data9/covid_detector_jpgs/cam/jpgs',
                        help='Cam output path')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args
def deprocess_image(img,mask=None):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img[mask==0]=0.5
    img = np.clip(img, 0, 1)

    return np.uint8(img*255)
def show_cam_on_image(img, mask,extral=None):
    if isinstance(extral,np.ndarray):
        mask=mask*extral[:,:,1]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam[extral==0]=np.float32(img)[extral==0]
    #cv2.imwrite("cam.jpg", np.uint8(255 * cam))
    return np.uint8(255 * cam)
def model_get(path):
    model = resnet152_plus(3)

    pretrained_dict = torch.load(path)
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    #print('matched keys:', len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
import glob
if __name__ == '__main__':
    args = get_args()
    target =args.image_path.split('raw_')[-1]
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model=model_get(args.model_path), \
                       target_layer_names=["6"], use_cuda=args.use_cuda,use_plus=True)
    gb_model = GuidedBackpropReLUModel(model=model_get(args.model_path), use_cuda=args.use_cuda,use_plus=True)
    o_path = args.output_path
    o_img_nii='/mnt/data9/covid_detector_jpgs/cam/pre'
    o_msk_nii = '/mnt/data9/covid_detector_jpgs/cam/mask'
    o_lung_nii='/mnt/data9/covid_detector_jpgs/cam/lung'
    i_path = args.mask_path
    i_path2 = args.image_path

    os.makedirs(o_path,exist_ok=True)
    os.makedirs(o_img_nii, exist_ok=True)
    os.makedirs(o_msk_nii, exist_ok=True)
    os.makedirs(o_lung_nii, exist_ok=True)
    lists=os.listdir(i_path)
    random.shuffle(lists)
    for cnt,names in enumerate(lists):
        if cnt>500:
            break
        try:
            img = cv2.imread(os.path.join(i_path, names), 1)
            img_raw=cv2.imread(os.path.join(i_path2,names),1)
            raw_shape=(img.shape[1],img.shape[0])
            rate=raw_shape/np.array([224,224])
            img_raw=np.float32(cv2.resize(img_raw,(224,224)))/255
            img = np.float32(cv2.resize(img, (224, 224))) / 255
        except:
            print(os.path.join(i_path2,names))
            continue
        input = preprocess_image(img)
        if target=='covid':
            target_index = 2
        elif target=='cap':
            target_index=1
        else:
            target_index=0
        mask,pred = grad_cam(input, 2)
        if pred[0,target_index]<0.8:
            continue
        cam=show_cam_on_image(img_raw, mask)##                  cam: cam on image
        gb = gb_model(input, index=2)
        gb = gb.transpose((1, 2, 0))

        cam_mask = cv2.merge([mask, mask, mask])
        ##                       attention_area: binary mask
        #cam_gb = deprocess_image(cam_mask*gb)
        gbt=gb.copy()
        gb = deprocess_image(gb)##                              gb: guided backward map
        attention_area = (cam_mask*(np.abs(gb-128)/255)) > 0.7
        attention_area=attention_area*(np.abs(gb-128)>64)
        attention_area=attention_area[:,:,0]+attention_area[:,:,1]+attention_area[:,:,2]
        attention_area=(attention_area>=1).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        attention_area = cv2.erode(cv2.morphologyEx(attention_area, cv2.MORPH_CLOSE, kernel),kernel)
        lung_mask=cv2.dilate(img[:,:,2],kernel)
        attention_area=attention_area*lung_mask
        attention_area = np.stack([attention_area, attention_area, attention_area], -1)
        if np.sum(attention_area)<=100:
            continue
        #cam = show_cam_on_image(img_raw, mask)
        cam_gb = deprocess_image(cam_mask * gbt,attention_area)  ##guided gradcam
        img_raw=cv2.resize(img_raw,raw_shape)
        cam_mask=cv2.resize(cam_mask,raw_shape)
        cam_gb=cv2.resize(cam_gb,raw_shape)
        cam=cv2.resize(cam,raw_shape)
        attention_area=cv2.resize(attention_area,raw_shape)
        I = np.concatenate([img_raw*255,cam,cam_gb,attention_area*255],1)
        #I=cam
        output_name = target+'_'+names
        output_path = os.path.join(o_path, output_name)

        cv2.imwrite(output_path, I)
        #np.save(output_path,cam_mask)
        Inii = sitk.GetImageFromArray(img_raw[:,:,1]*255)
        Lnii = sitk.GetImageFromArray(img[:, :, 2])
        Mnii = sitk.GetImageFromArray(attention_area[:,:,1])
        sitk.WriteImage(Inii,os.path.join(o_img_nii,output_name[:-4]+'.nii'))
        sitk.WriteImage(Mnii,os.path.join(o_msk_nii, output_name[:-4]+ '.nii') )
        sitk.WriteImage(Lnii, os.path.join(o_lung_nii, output_name[:-4] + '.nii'))
