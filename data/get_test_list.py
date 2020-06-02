import os,glob,random
#data_path='/home/cwx/extra/covid_project_data'
#seg_path='/home/cwx/extra/covid_project_segs'
seg_path = '/mnt/data11/dr_ct_segs/CT/'
data_path = '/mnt/data11/dr_ct_data/CT/'

f2=open('train_xct2.list','w')
f=open('test_xct2.list','w')
def set_it(all_files,set_name):
    person_name=[item.split('/')[-1].split('_')[0]+'_'+item.split('/')[-1].split('_')[1] for item in all_files]
    #person_name = [item.split('/')[-1].split('_')[0] for item in all_files]
    person_name=list(set(person_name))
    l = len(person_name)
    for i,name in enumerate(person_name):
        if i<l//2:
            all_ct=glob.glob(os.path.join(data_path, set_name, name+'_*.nii'))
            for act in all_ct:
                f.writelines(act + ',' +
                             act.replace('_data','_segs/lungs')
                             .replace('/'+set_name+'/','/'+set_name+'/'+set_name+'_') + '\n')
        else:
            all_ct = glob.glob(os.path.join(data_path, set_name, name + '_*.nii'))
            for act in all_ct:
                f2.writelines(act + ',' + act.replace('_data', '_segs/lungs')
                              .replace('/' + set_name + '/', '/' + set_name + '/' + set_name+'_') + '\n')


for set_name in os.listdir(data_path):
    all_files=glob.glob(os.path.join(data_path,set_name,'*.nii'))
    random.shuffle(all_files)
    set_it(all_files,set_name)


