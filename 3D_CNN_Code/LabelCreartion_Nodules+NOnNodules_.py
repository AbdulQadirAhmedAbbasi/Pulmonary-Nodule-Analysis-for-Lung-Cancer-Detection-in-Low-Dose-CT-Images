#=======================================

#Project : Lung Nodule Detection 
#Student Name: Abdul Qadir Ahmed Abbasi
#Supervisor: Dr. Hafeez Ur Rehman
#Label Creration for Both Nodule & Non Nodules

#=======================================

import csv

#defining paths for both nodules & non nodules
positive_cand_path = 'D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/patches3D/positiveLabels3D.csv'
negative_cand_path = 'D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/patches3D/negativeLabels3D.csv'

#function for reading CSV
def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

#loading data
positive_cands = readCSV(positive_cand_path)
negative_cands = readCSV(negative_cand_path)

#function for writing to CSV file
def csv_writer(data, path, handler):
    """
    Write data to a CSV file path
    """
    writer = csv.writer(handler, delimiter=',')
    writer.writerow(data)

#defining path for a single file which will have labels of both nodules & non nodules	
labels_path = "D:/FDrive/StudyStuff/Semester7/FYP-1/DataSet/patches3D/labels3D.csv"



#label creartion starts here
k = 0
with open(labels_path, "w", newline='') as csv_file:
    data = ['patch_ID', 'nodule']
    csv_writer(data, labels_path, csv_file)
    for j in range(48):     
        for i in range(1184):
    
            patch_ID = positive_cands[i*48+j][0]
            nodule = str(positive_cands[i*48+j][1])
            data = [patch_ID, nodule]
            csv_writer(data, labels_path, csv_file)
        
            patch_ID = negative_cands[k][0]
            nodule = str(negative_cands[k][1])
            data = [patch_ID, nodule]
            csv_writer(data, labels_path, csv_file)
            k = k + 1
        
csv_file.close()
print(k)
print(i*48+j)
print('Labels Creation Completed !!')	
	