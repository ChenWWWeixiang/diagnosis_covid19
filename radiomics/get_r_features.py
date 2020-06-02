from radiomics import featureextractor
import os,csv,six
import numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-o", "--outputfile", help="output file's name", type=str,
                    default='r_features.csv')
parser.add_argument("-m", "--inputmask", help="input mask root", type=str,
                    default='/mnt/data9/cam/mask/')
parser.add_argument("-r", "--inputimgs", help="input data root", type=str,
                    default='/mnt/data9/cam/pre/')
args = parser.parse_args()

o_img_nii = args.inputimgs
o_msk_nii = args.inputmask
extractor = featureextractor.RadiomicsFeatureExtractor('RadiomicsParams.yaml')
with open(args.outputfile, 'w+', newline='') as f:
    writer = csv.writer(f)
    for i,name in enumerate(os.listdir(o_img_nii)):
        row = ['id', 'label']
        if 'cap' in name:
            cls=1
        elif 'AB-in' in name:
            cls=0
        else:
            cls=2
        row_next = [name, cls]
        imageName=os.path.join(o_img_nii,name)
        maskName=os.path.join(o_msk_nii,name)
        try:
            result = extractor.execute(imageName, maskName)
            for idx, (key, val) in enumerate(six.iteritems(result)):
                if idx<11:
                    continue
                if not isinstance(val,(float,int,np.ndarray)):
                    continue
                if np.isnan(val):
                    val=0
                   # print(val)
                row.append(key)
                row_next.append(val)
            if i ==0:
                writer.writerow(row)
            writer.writerow(row_next)
        except:
            print(imageName)
            os.remove(imageName)
            os.remove(maskName)