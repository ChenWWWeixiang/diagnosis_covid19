import os,random,glob
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-p", "--path", help="A list of paths to jpgs for seperate",
                    type=str,
                    default= '/mnt/data9/independent_raw/')
parser.add_argument("-t", "--train_txt",
                    help="train list output path",
                    type=str,
                    default='txt/ind_train_jpg.txt')
parser.add_argument("-v", "--val_txt",
                    help="validation list output path",
                    type=str,
                    default='lists/ind_list.list')

args = parser.parse_args()
#path=['/mnt/data7/slice_test_seg/jpgs2']
f1 = open(args.train_txt, 'w')
f2 = open(args.val_txt, 'w')
path=args.path
c=0
train_jpg=[]
test_jpg=[]
for type in os.listdir(path):
    All = []
    for person in os.listdir(os.path.join(path,type)):
        for scan in os.listdir(os.path.join(path,type,person)):
            jpgs=os.listdir(os.path.join(path,type,person,scan))
            id=[int(j.split('.')[0]) for j in jpgs]
            try:
                mm=np.max(id)
                valid=np.arange(int(mm//4),int(mm*3//4)).tolist()
                jpgs=[os.path.join(path,type,person,scan,j) for j in jpgs if int(j.split('.')[0]) in valid]
                All+=jpgs
            except:
                continue
    persons=[allone.split('/')[-3] for allone in All]
    persons=list(set(persons))
    random.shuffle(persons)
    ptrain=persons[:len(persons)//2]
    ptest = persons[len(persons) // 2:]
    train_jpg+=[a for a in All if a.split('/')[-3] in ptrain]
    for item in ptest:
        ap=os.listdir(os.path.join(path, type, item))
        test_jpg+=[os.path.join(path,type,item,p)for p in ap]

for p in train_jpg:
    f1.writelines(p+'\n')
for p in test_jpg:
    f2.writelines(p+'\n')

