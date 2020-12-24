#=======================================

#Project : Lung Nodule Detection 
#Student Name: Abdul Qadir Ahmed Abbasi
#Supervisor: Dr. Hafeez Ur Rehman
#Non Nodule Patch Extraction 

#=======================================

import SimpleITK as sitk
import csv
import sys
import math
import pickle
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage
import os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from decimal import Decimal, ROUND_CEILING
from random import randint
import scipy.misc
import matplotlib.pyplot as plt
%matplotlib inline

#function for loading CT scan

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing
	
	
dir_path  = 'D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/CTScans/'
cand_path = 'D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/candidates_for_negative_patch_extraction.csv'


#function for normalization
def normalizePlanes(npzarray):
     
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

#function for reading CSV
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


cands = readCSV(cand_path)

#function for changing world coordinates to voxel coordinates
def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

#function for writing to CSV
def csv_writer(data, path, handler):
    """
    Write data to a CSV file path
    """
    writer = csv.writer(handler, delimiter=',')
    writer.writerow(data)
        
labels_path = "D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/patches3D/negativeLabels3D.csv"


#patch extraction and label creation starts here for nodules only
counter = 0
check = 0
with open(labels_path, "w", newline='') as csv_file:
    data = ['patch_ID', 'nodule']
    csv_writer(data, labels_path, csv_file)
    for cand in cands:
        if(check == 0):
            img_path = dir_path + cand[0] + '.mhd'
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
            worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
            voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
            check = check + 1
            
        elif(cand[0] == previous_cand):
            pass
            #print("I m called")
        else:
            img_path = dir_path + cand[0] + '.mhd'
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
 
            
        previous_cand = cand[0]
        
        worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
        voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
        
        #3D patch is of size 32 * 32 * 32 in z,y,x dimensions 
        voxelWidth = 32
        patch = numpyImage[int(voxelCoord[0]-voxelWidth/2):int(voxelCoord[0]+voxelWidth/2),int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2),int(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2)]
        if(patch.shape == (32,32,32)):
            if(cand[4] == str(0)):
                patch = normalizePlanes(patch)
                outputDir = 'D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/patches3D/original/both'

                str0 = str(worldCoord[0])
                str1 = str(worldCoord[1])
                str2 = str(worldCoord[2])
                nodule = cand[4]                      
                final_patch_path = os.path.join(outputDir, 'patch_' + str0 + '_' + str1 + '_' + str2)
                #saving as numpy arrays
                np.save(final_patch_path + '.npz', patch)
                #writing labels to csv file
                patch_ID = 'patch_' + str0 + '_' + str1 + '_' + str2
                data = [patch_ID, nodule]
                csv_writer(data, labels_path, csv_file)
                counter = counter + 1
            
        if(counter == 56832):
            break 
        
csv_file.close()
print("Done !")	