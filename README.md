# OpenCovidDetector: Open source algorithm for detecting COVID-19 on Chest CT


OpenCovidDetector is an opensource COVID-19 diagnosis system implementing on pytorch, which is also 
as presented in our paper: Development and Evaluation of an AI System for COVID-19 Diagnosis
 (https://www.medrxiv.org/content/10.1101/2020.03.20.20039834v2)
 

About the Project
------
Early detection of COVID-19 based on chest CT will enable timely treatment of patients and help control the spread of the disease. With rapid spreading of COVID-19 in many countries, however, CT volumes of suspicious patients are increasing at a speed much faster than the availability of human experts. We proposed an artificial intelligence (AI) system for fast COVID-19 detection and performed extensive statistical analysis of CTs of COVID-19 based on the AI system. We developed and evaluated our system on a large dataset with more than 10 thousand CT volumes from COVID-19, influenza-A/B, non- viral community acquired pneumonia (CAP) and non-pneumonia subjects. In such a difficult multi-class diagnosis task, our deep convolutional neural network-based system is able to achieve an area under the receiver operating characteristic curve (AUC) of 97.17%, a sensitivity of 90.19%, and a specificity of 95.76% for COVID-19 on internal test cohort of 3,203 scans and AUC of 97.77% on the publicly available CC-CCII database with 1,943 test samples. In a reader study involving five radiologists, the AI system outperforms all of radiologists in more challenging tasks at a speed of two orders of magnitude above them. Diagnosis performance of chest x-ray (CXR) is compared


Methods
----------
 We proposed a deep-learning based AI system for COVID-19 diagnosis, which directly takes CT data as input, performs lung segmentation, COVID-19 diagnosis and COVID-infectious slices locating. In addition, we hope that the diagnosis results of AI system can be quantitatively explained in the original image to alleviate the drawback of deep neural networks as a black box.
 
 ![image](https://github.com/ChenWWWeixiang/diagnosis_covid19/blob/master/pic/fw.jpg)
 
 We developed and evaluated a deep learning-based COVID-19 diagnosis system, using multi-class multi-center data, which includes 10,250 CT scans from 7,917 subjects consisting of COVID-19, CAP, influenza and non-pneumonia. CAP subjects included in our database were all non-viral CAP. Data were collected in three different centers in Wuhan, and from three publicly available databases, LIDC-IDRI, Tianchi-Alibaba, and CC-CCII.
 

 Results 
----------
 ![image](https://github.com/ChenWWWeixiang/diagnosis_covid19/blob/master/pic/roc_4c.jpg)

The trained AI system was evaluated on the test cohort. We used the receiver operating characteristic (ROC) curves to evaluate the diagnostic accuracy. On the test cohort, the ROC curve showed AUC of four categories were respectively 0.9752 (for non-pneumonia), 0.9804 (for CAP), 0.9885 (for influenza) and 0.9745 (for COVID-19). Besides, sensitivity and specificity of COVID-19 were 0.9019 and 0.9576.
  
In the reader study, the diagnostic accuracy of the AI system outperformed experienced radiologists in two tasks from the outbreak center, with AUC of 0.9869, 0.9727, 0.9585 separately for pneumonia-or-non-pneumonia, CAP-or-COVID-19 and influenza-or-COVID-19 tasks.
 
  ![image](https://github.com/ChenWWWeixiang/diagnosis_covid19/blob/master/pic/roc_cap_covid.jpg)
  
    ![image](https://github.com/ChenWWWeixiang/diagnosis_covid19/blob/master/pic/roc_influenza_covid.jpg)
    
   CXR is also considered as a possible way to diagnose COVID-19. And after using t-SNE, we found that COVID-19 subjects were mapped to more than one clusters.Samples in left cluster of COVID-19 were most in early and mild stage which have small GGO with nearly round shape. Samples in right cluster had larger lesion and some of them had crazy paving patterns. Fibration and consolidation could be found in the upper cluster whose sizes of lesion were generally between lefts and rights. Although visualization by t-SNE was a conjecture for extracting features from the network, we can clearly find that patients of COVID-19 may be divided into different subclasses.
   ![image](https://github.com/ChenWWWeixiang/diagnosis_covid19/blob/master/pic/roc_xct.jpg)
   
   ![image](https://github.com/ChenWWWeixiang/diagnosis_covid19/blob/master/pic/t-SNE-map.jpg)
Guidance to Use
-------
###  Environment
The code has been succesfully run in Ubuntu 16.04, python 3.6.1,CUDA 10.0.

Packages requirements:
matplotlib==3.1.2
six==1.13.0
torch==1.3.1
scikit_image==0.16.2
imageio==2.6.1
scipy==1.3.3
numpy==1.15.3
opencv_python==4.1.1.26
pandas==0.23.4
torchvision==0.4.2
Pillow==6.2.1
pydicom==1.4.2
pyradiomics
scikit_learn==0.22.2.post1
seaborn==0.10.0
SimpleITK==1.2.4
skimage
toml==0.10.0
xlrd==1.2.0

run ```pip install -r requirements.txt``` to install all above packages.

Usage
----
1. Lung area segmentation: prepare your own data in nii format. Run ```segmentation/predict.py``` to get segmentations.
2. Get train list and test list: Run ```data/get_test_list.py``` to divide dataset into training and test.
3. Get training jpgs: Run ```data/get_train_jpgs.py``` to extract jpgs from training cohort.
4. Get training jpg list: Run ```data/get_set_seperate_jpg.py``` to get the training file lists for training.
5. Start training diagnosis net: run ```main.py``` to start training and test. Training parameters are listed in options_lip.toml.
6. Evaluate diagnosis net:run ```testengine.py``` to test. Test parameters are listed in test.toml.

*. Figure plting tools: in result_plt.
*. Radiomics tools: in radiomics.

Citation
----

If you find this project helpful, please cite our paper:
```
@article {OpenCovidDetector,
	author = {Jin, Cheng and Chen, Weixiang and Cao, Yukun and Xu, Zhanwei and Zhang, Xin and Deng, Lei and Zheng, Chuansheng and Zhou, Jie and Shi, Heshui and Feng, Jianjiang},
	title = {Development and Evaluation of an AI System for COVID-19 Diagnosis},
	year = {2020}, doi = {10.1101/2020.03.20.20039834},journal = {medRxiv}}
```
