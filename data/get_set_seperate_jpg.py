import os,random,glob
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-p", "--path", help="A list of paths to jpgs for seperate",
                    type=str,
                    default=['/mnt/data9/m_leision_jpgs_train',
                            #'/mnt/data9/covid_detector_jpgs/selected_train2/nor',
                            #'/mnt/data9/covid_detector_jpgs/selected_train2/abnor',
                            #'/mnt/data9/covid_detector_jpgs/selected_train1/nor',
                            # '/mnt/data9/covid_detector_jpgs/selected_train1/abnor',
                             ])
parser.add_argument("-t", "--train_txt",
                    help="train list output path",
                    type=str,                    
                    default='txt/train_slice2.txt')
parser.add_argument("-v", "--val_txt",
                    help="validation list output path",
                    type=str,
                    default='txt/1.txt')

args = parser.parse_args()
if isinstance(args.path,str):
    path=eval(args.path)
else:
    path=args.path
#path=['/mnt/data7/slice_test_seg/jpgs2']
f1 = open(args.train_txt, 'w')

c=0
for ipath in path:
    cnt = 0
    files=os.listdir(ipath)
    if len(files)==0:
        continue
    names_id=[file.split('_')[0]+'_'+file.split('_')[1]+'_'+file.split('_')[2] for file in files]
    names_id=list(set(names_id))
    set_name=files[0].split('_')[0]
    random.shuffle(names_id)
    train=names_id
    #val=names_id[len(names_id)//2:-len(names_id)//4]
    #test=names_id[-len(names_id)//4:]
    for _,i in enumerate(train):
        names=glob.glob(ipath+'/'+i+'_*')
        for name in names:
            cnt+=1
            c+=1
            f1.writelines(name+'\n')
#print(c)
    #for i in test:
    #    names = glob.glob(ipath + '/' + set_name + '_' + i + '_*')
    #    for name in names:
    #        f2.writelines(name+'\n')