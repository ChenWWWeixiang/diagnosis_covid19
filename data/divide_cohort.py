import os,glob,random
ct_x_root='/home/cwx/extra/dr_ct_data/CT'
all_ct_root='/home/cwx/extra/covid_project_data'
xct_seg='/mnt/data11/seg_of_XCT/lung'
x_train_list=[]
x_test_list=[]
if True:
    with open('testlist_xct.list', 'r') as f:
        x_test_list=f.readlines()
        x_test_list=[da.split(',')[0] for da in x_test_list]
else:
    for subdir in os.listdir(ct_x_root):
        ct_names=glob.glob(os.path.join(ct_x_root,subdir,'*.nii'))
        person=[da.split('/')[-1].split('_')[0]+'_'+da.split('/')[-1].split('_')[1] for da in ct_names]
        person=list(set(person))
        random.shuffle(person)
        if subdir == 'covid':
            person_d=[da for da in person if da.split('_')[0]=='10']
            train_p = person_d[:len(person_d)//2]
            test_p = person_d[len(person_d) // 2:]
           # print(len(person_d)//2)
            person_d=[da for da in person if da.split('_')[0]!='10']
            train_p += person_d[:len(person_d)//2]
            test_p += person_d[len(person_d) // 2:]
        else:
            train_p = person[:len(person)//2]
            test_p = person[len(person) // 2:]
        for item in train_p:
            x_train_list+=glob.glob(os.path.join(ct_x_root,subdir,item+'_*.nii'))
        for item in test_p:
            x_test_list+=glob.glob(os.path.join(ct_x_root,subdir,item+'_*.nii'))

    with open('trainlist_xct.list','w') as f:
        for item in x_train_list:
            set_name=item.split('/')[-2]
            f.writelines(item + ',' +
                         os.path.join(xct_seg,set_name,set_name+'_'+item.split('/')[-1]) + '\n')
    with open('testlist_xct.list','w') as f:
        for item in x_test_list:
            set_name=item.split('/')[-2]
            f.writelines(item + ',' +
                         os.path.join(xct_seg,set_name,set_name+'_'+item.split('/')[-1]) + '\n')

ct_train_list=[]
ct_test_list=[]
for set_name in os.listdir(all_ct_root):
    all_files=glob.glob(os.path.join(all_ct_root,set_name,'*.nii'))
    person_name = [item.split('/')[-1].split('_')[0] + '_' + item.split('/')[-1].split('_')[1] for item in
                   all_files]
    person_name = set(person_name)
    num_person = len(person_name)
    if  set_name=='cap-zs':
        #continue
        must_test_name=[item.split('/')[-1].split('_')[0] + '_' + item.split('/')[-1].split('_')[1] for item in
                        x_test_list if 'CAP' in item]
    elif set_name=='covid2':
        must_test_name = [item.split('/')[-1].split('_')[0] + '_' + item.split('/')[-1].split('_')[1] for item in
                          x_test_list if 'covid' in item and item.split('/')[-1].split('_')[0]=='10']
    elif set_name=='covid':
        #continue
        must_test_name = [item.split('/')[-1].split('_')[0] + '_' + item.split('/')[-1].split('_')[1] for item in
                          x_test_list if 'covid' in item and item.split('/')[-1].split('_')[0] != '10']
    else:
        #continue
        must_test_name=[]
    must_test_name = set(must_test_name)
    to_test=list(person_name.intersection(must_test_name))
    left=list(person_name.difference(to_test))

    #assert len(left)>=num_person//2
    to_test_num=max(num_person//2-len(to_test),0)
    random.shuffle(left)
    to_test+=left[:to_test_num]
    to_train=left[to_test_num:]
    for item in to_test:
        ct_train_list += glob.glob(os.path.join(all_ct_root, set_name, item + '_*.nii'))
    for item in to_train:
        ct_test_list += glob.glob(os.path.join(all_ct_root, set_name, item + '_*.nii'))

with open('trainlist_ct_only.list','w') as f:
    for item in ct_train_list:
        set_name=item.split('/')[-2]
        f.writelines(item + ',' +
                     item.replace('_data', '_segs/lungs')
                     .replace('/' + set_name + '/', '/' + set_name + '/' + set_name + '_') + '\n')
with open('testlist_ct_only.list','w') as f:
    for item in ct_test_list:
        set_name=item.split('/')[-2]
        f.writelines(item + ',' +
                     item.replace('_data', '_segs/lungs')
                     .replace('/' + set_name + '/', '/' + set_name + '/' + set_name + '_') + '\n')
