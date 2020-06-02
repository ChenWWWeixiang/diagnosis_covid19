import numpy as np
import os
from tqdm import tqdm
import torch,cv2,shutil,glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from segmentation.unet import UNet
import SimpleITK as sitk
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def get_model(model_path, n_classes=3, cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.set_grad_enabled(False)
    model = UNet(n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def predict(img, model, batch_size=4, lesion=False, cuda=True):
    if isinstance(model, str):
        model = get_model(model)
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    img = sitk.GetArrayFromImage(img)
    img[img < -1024] = -1024
    img[img > 500] = 655
    if lesion:
        img = (img + 1024) / 1624.
    else:
        img = img / 255.
    # print(model)

    data = torch.from_numpy(img[:, np.newaxis, :, :])
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    _, H, W = img.shape

    res = np.zeros((1, 512, 512), dtype=np.int8)
    for i, data in enumerate(loader):
        if not H == 512:
            data[0] = torch.nn.functional.interpolate(data[0], (512, 512))
        images = data[0].to(device)
        images = images.float()
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        labels = labels.cpu().numpy()
        res = np.concatenate((res, labels), axis=0)

    return res[1:, :, :]


if __name__ == '__main__':
    clss=['NCP','CP','Normal']
    model_path = './lung_checkpoint.pth'
    model1 = get_model(model_path)
    model2 = './checkpoint_final.pth'
    model2 = get_model(model2, n_classes=2)
    print('get model done')
    for cls in clss:
        outpath='/mnt/data9/independent_data/'+cls
        os.makedirs(outpath,exist_ok=True)
        root='/mnt/data9/independent_raw/'+cls
        for patient in os.listdir(root):
            for scan in os.listdir(os.path.join(root,patient)):
                if os.path.exists(os.path.join(outpath,patient+'_'+scan+'.nii')):
                    continue
                I=[]
                for slice in os.listdir(os.path.join(root,patient,scan)):
                    img=cv2.imread(os.path.join(root,patient,scan,slice))[:,:,1]
                    img=cv2.resize(img,(512,512))
                    img=img/255.0*1500-1400
                    I.append(img)
                I=np.array(I)
                try:
                    I=sitk.GetImageFromArray(I)
                    sitk.WriteImage(I,os.path.join(outpath,patient+'_'+scan+'.nii'))
                    print(os.path.join(outpath,patient+'_'+scan+'.nii'))
                except:
                    continue

        img_path = '/mnt/data9/independent_data/'+cls
        pred_path = '/mnt/data9/independent_segs/lungs/'+cls
        jpg_path = '/mnt/data9/independent_crop/' + cls
        os.makedirs(jpg_path,exist_ok=True)
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        print('# images:  ', len(os.listdir(img_path)))

        for filename in tqdm(os.listdir(img_path), dynamic_ncols=True):
            #print(filename)
            if os.path.exists(os.path.join(pred_path, pred_path.split('/')[-1] + '_' + filename)):
                continue
            try:
                img = sitk.ReadImage(os.path.join(img_path, filename))
            except:
                continue
            result = predict(img,model1,batch_size=15)
            result[result > 1] = 1
            result = np.array(result, np.uint8)
            result = sitk.GetImageFromArray(result)
            sitk.WriteImage(result, os.path.join(pred_path, pred_path.split('/')[-1] + '_' + filename))
            a = 1
            result=sitk.GetArrayFromImage(result)
            img = sitk.GetArrayFromImage(img)
            for i in range(result.shape[0]):
                I=img[i,:,:]
                M=result[i,:,:]
                I=(I+1400)/1500*255
                I=np.stack([I,I,M*255],-1)
                img_out_path=os.path.join(jpg_path,filename.split('.')[0]+'_'+str(i)+'.jpg')
                cv2.imwrite(img_out_path,I)



