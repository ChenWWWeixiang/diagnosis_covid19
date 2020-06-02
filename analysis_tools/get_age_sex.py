import pydicom,os
import numpy as np
import csv
from pydicom.data import get_testdata_files

Ages=[]
Sexes=[]
cnt=0
nn=0
#root='/mnt/data7/resampled_data/'
ill_root = '/home/cwx/extra/NCP_CTs/'
control_root = '/home/cwx/extra/NCP_CTs/NCP_control/control'
f=open('raw_ages.txt','w')
for set_name in os.listdir(ill_root):
    if set_name=='NCP_control' or set_name[:3]!='NCP':
        continue
    for person in os.listdir(os.path.join(ill_root,set_name)):
        for phase in os.listdir(os.path.join(ill_root,set_name,person)):
            if not os.path.isdir(os.path.join(ill_root,set_name,person, phase)):
                continue
            for inner in os.listdir(os.path.join(ill_root,set_name,person, phase)):
                if not os.path.isdir(os.path.join(ill_root,set_name,person, phase, inner)):
                    continue
                for innner in os.listdir(os.path.join(ill_root, set_name,person, phase,inner)):
                    if not os.path.isdir(os.path.join(ill_root, set_name,person, phase, inner,innner)):
                        continue
                    adicom = os.listdir(os.path.join(ill_root,set_name,person, phase, inner,innner))

                    adicom = [a for a in adicom if a[0] == 'I']

                    adicom = adicom[0]

                    # print(os.path.join(root, patient, case, phase, inner, adicom))
                    ds = pydicom.read_file(os.path.join(ill_root,set_name,person, phase, inner,innner, adicom))
                    try:
                        age = int(ds['PatientAge'].value[:-1])
                        sex = ds['PatientSex'].value
                    except:
                        try:
                            age = 2020-int(ds['PatientBirthDate'].value[:4])

                        except:
                            age=45
                        sex = ds['PatientSex'].value
                    f.writelines(set_name+'/'+person+'\t'+str(age)+'\t'+str(sex)+'\n')
                    print(set_name+'/'+person+','+str(age) + ',' + str(sex))
                    # cnt+=1
                    break
                break
            break
f.close()