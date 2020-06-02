import os,random
data_path='/home/cwx/extra/covid_project_data'
seg_path='/home/cwx/extra/covid_project_segs'
nums=[20,30]
sets=['HxNx','covid']
with open('reader_HN_vs_covid.list','w') as f:
    ALL=[]
    with open('3cls_test2.list','r') as l:
        data=l.readlines()
        for set_this,num_this in zip(sets,nums):
            this_data_t=[da.split(',')[0] for da in data if set_this+'/' in da]
            this_data=[da.split('/')[-1].split('_')[0]+'_'+da.split('/')[-1].split('_')[1] for da in this_data_t]
            this_data=list(set(this_data))
            random.shuffle(this_data)
            selected=this_data[:num_this]
            for one in selected:
                find_name=set_this+'/'+one
                print(find_name)
                find_items=[da for da in data if find_name in da]
                ALL.append(find_items[0])
    random.shuffle(ALL)
    for i in range(len(ALL)):
        f.writelines(ALL[i])
