#!/usr/bin/env bash
#/home/cwx/anaconda3/envs/densenet.pytorch/bin/python testengine.py
cd data
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python get_set_seperate_jpg.py
cd ..
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python main.py
cd multi_period_scores
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python analyse_slice.py >> 1.txt
cd ../data
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python get_set_seperate_jpg.py
cd ..
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python main.py
cd multi_period_scores
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python analyse_slice.py >> 2.txt
cd ../data
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python get_set_seperate_jpg.py
cd ..
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python main.py
cd multi_period_scores
/home/cwx/anaconda3/envs/densenet.pytorch/bin/python analyse_slice.py >> 3.txt