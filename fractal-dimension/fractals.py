import cv2
import sys
import numpy as np
from functools import reduce
from PIL import Image
from math import floor, log, ceil # the whole house
from matplotlib import pyplot as plt
import SimpleITK as sitk


# convert colour image to black and white
def convert_to_blacks(img):
    thresh = cv2.THRESH_BINARY
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bw = cv2.threshold(img_gray, 127, 255, thresh)
    return img_bw


# take an even spread of num from a list
#this is definitely not borrowed from stackoverflow
def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]


# returns factors of n    
#I definitely borrowed this from stackoverflow
def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


# check if box is touching fractal
def check_black(b, i, k, arr):
    thresh = b*b*1
    return arr[i:i+b,k:k+b].sum() > thresh


#supply a blocksize and a matrix to get the hitcount
def hits_with_boxsize(arr, boxsize):
    shape = arr.shape 
    results = np.zeros([ int(shape[0]/boxsize+0.5) , int(shape[1]/boxsize+0.5) ])
    
    # for each box
    for i in range(0, shape[0], boxsize):
        for k in range(0, shape[1], boxsize):
            box_x = floor(i/boxsize)
            box_y = floor(k/boxsize)
            
            #check inside that box
            if(check_black(boxsize, i, k, arr)):
                results[box_x, box_y] = 1
    hits = results.sum()
    return (hits, results) # why return two results when one is derived from the other?

import os
import argparse
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--outputfile", help="output file's name", type=str,
                        default='../HFD.txt')
    parser.add_argument("-i", "--inputfile", help="inputfile root", type=str,
                        default='/mnt/data9/cam/mask/')
    args = parser.parse_args()
    filename = args.inputfile
    f=open(args.outputfile,'w')
    f.writelines('name' + ',' + 'value' + '\n')
    for item in os.listdir(filename):
        #img = Image.open(filename).convert('RGB')
        data=sitk.ReadImage(os.path.join(filename,item))
        img=sitk.GetArrayFromImage(data)*255
        arr = np.array(img, dtype=np.uint8)
        arr = np.invert(arr, dtype=np.uint8)
        arr=cv2.resize(arr,(255,255))
        #cv2_img = cv2.imread(filename, 0)
        #img_bw = convert_to_blacks(arr)

        if(arr.shape[0] != arr.shape[1]):
            print("dimensions aren't square... there will likely be an error")

        fact = sorted(factors(len(arr))) # get all factors of image size
        spaced10 = fact[1:len(fact)-1] #remove 1 and n.
        if(len(fact) > 10): # could be lots of factors, only take 10
            spaced10 = list(takespread(spaced10, 10))  # an even distribution to view the fractal at different sizes

        hits, box_arr = zip(*list(hits_with_boxsize(arr, i) for i in spaced10)) # get box-counting results

        dim = list( (log(hits[i]/hits[i+1], spaced10[i+1]/spaced10[i]) for i in range(len(hits)-1))) # calculate dimension
        #print("Fractal dimension measurement taken at different box sizes.")
        dim=np.min(dim)
        print(item,dim)
        f.writelines(item+','+str(dim)+'\n')
    f.close()
main()
