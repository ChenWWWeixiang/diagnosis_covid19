import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
all_files=os.listdir('npys')
def ana_group(name,group):
    R=[]
    for item in group:
        data=np.load(os.path.join('npys_re',item))
        ab_rate=np.mean(data>0.5)
        R.append(ab_rate)
    print('subset:' ,name,', # abnormal slice / # all slice: ',np.mean(R),', #subjects: ',len(R))
    return R
def show_bins(data,name,fig_name):
    bin_edges=np.arange(0,1.1,0.1)
    num_g=len(data)
    bin_width = bin_edges[1] - bin_edges[0]
    x = bin_edges[:-1]
    width = bin_width / num_g
    plt.figure()
    for i in range(num_g):
        hist, _ = np.histogram(data[i],bins=bin_edges)
        hist=hist/hist.sum()
        plt.bar(x+width/2+i*width,hist,width,alpha = 0.9,label=name[i])
    plt.xlim([0,1])
    plt.xlabel('Ratio of Abnormal Slices to All Slices')
    plt.ylabel('Pecentage')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()


male=[da for da in all_files if 'M' in da]
female=[da for da in all_files if 'F' in da]
age=[int(da.split('_')[-2])//20 for da in all_files]
age=np.array(age)
r1=ana_group('Male',male)
r2=ana_group('Female',female)
show_bins([r1,r2],['Male','Female'],'jpgs/gender.jpg')


age4=np.array(all_files)[age==4]
age1=np.array(all_files)[age==1]
age2=np.array(all_files)[age==2]
age3=np.array(all_files)[age==3]

r1=ana_group('21-40',age1)
r2=ana_group('41-60',age2)
r3=ana_group('61-80',age3)
r4=ana_group('80+',age4)
show_bins([r1,r2,r3,r4],['21-40','41-60','61-80','80+'],'jpgs/age.jpg')
STAGEI = []
STAGEII = []

person = [item.split('_')[0] + '_' +
          item.split('_')[1] for item in all_files]
time = [int(item.split('_')[-3][-4:]) for item in all_files]
time = np.array(time)
person = np.array(person)
unit_person = list(set(person))
unit_person = np.array(unit_person)
cnt2 = 0
for iperson in unit_person:
    this_idx = np.where(person == iperson)[0]
    this_time = time[this_idx]
    if len(this_time) >= 2:
        cnt2 += 1
    else:
        continue
    sorted_idx = this_idx[np.argsort(this_time)]
    STAGEI.append(all_files[sorted_idx[0]])
    STAGEII.append(all_files[sorted_idx[1]])
r1=ana_group('Stage I',STAGEI)
r2=ana_group('Stage II',STAGEII)
show_bins([r1,r2],['stage1','stage2'],'jpgs/stage.jpg')