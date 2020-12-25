#=======================================

#Project : Lung Nodule Detection 
#Student Name: Abdul Qadir Ahmed Abbasi
#Supervisor: Dr. Hafeez Ur Rehman
#3D CNN Architecture Keras & Training + Testing

#=======================================

#Importing Liabraries

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix


from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2

import tensorflow
import theano
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing
%matplotlib inline

#Defining Paths for 3D Patches & Labels of Both Nodules & Non Nodules

data_dir = 'myDataSet/patches3D/original/both/'
labels_path = 'myDataSet/patches3D/labels.csv'

#Function for Reading CSV File

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

#Reading Labels in Single Variable	 & Prinintg Details

nodules = readCSV(labels_path)

num_samples = len(nodules)
print(num_samples)  #One extra due to heading of Patch ID and Nodules Labels, total patches are Total minus 1

print(nodules[0]) # printing first array index 
print(nodules[1]) #testing
print(nodules[1][0])#testing
print(nodules[1][1])#testing


# image specification
img_rows,img_cols,img_depth=32,32,32

# Training data
X_tr=[]           # variable to store entire dataset

#Assign Label to each class
label=np.ones(((num_samples-1),),dtype = int)
i = 0                # for indexing label array

for nodule in nodules[1:]:
    
    noduleID = nodule[0]
    nodule_label = nodule[1]
    #print(noduleID, nodule_label)

    #complete path of one 3D patch
    path = data_dir + noduleID + '.npz.npy'

    #array of pixels of patch
    patch_data = np.load(path)
    #print(patch_data.shape)
    
    #inverting axis of array from z,y,x to x,y,z
    patch_data = np.rollaxis(np.rollaxis(patch_data,2,0),2,0)
    #print(patch_data.shape)
    
    #appending all patches
    X_tr.append(patch_data)
    label[i] = nodule_label
    
    i = i + 1
	
	
	
X_tr_array = np.array(X_tr)   # convert the frames read into array
num_samples = len(X_tr_array) 
print(num_samples)

num_samples_label = len(label) 
print(num_samples_label)


#complete training data with patches and labels
train_data = [X_tr_array,label]
print(train_data[0].shape)

#spliting training data and training labels
(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)	

#Making 5D input for Conv3D
train_set = np.zeros((num_samples, 1, img_rows, img_cols,img_depth))

for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]
    
patch_size = 7    # img_depth

print(train_set.shape, 'train samples')

#inverting axis 
train_set = np.rollaxis(np.rollaxis(train_set,2,1), 1,2)
print(train_set.shape)

train_set = np.rollaxis(np.rollaxis(train_set,3,2), 2,3)
print(train_set.shape)

train_set = np.rollaxis(np.rollaxis(train_set,4,3), 3,4)
print(train_set.shape)



# CNN Training parameters

batch_size = 2
nb_classes = 2
nb_epoch = 10

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

# Pre-processing
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)


#Defining Model (Reference Research Paper For Model)

model = Sequential()

model.add(Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same', input_shape=(32,32,32,1)))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1)))

model.add(Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1)))

model.add(Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(1,1,1)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('relu'))

model.compile(loss='binary_crossentropy', optimizer= 'Adadelta', metrics = ['accuracy'])

print('Done.')


# Split the data into test & train sets

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.30, random_state=42)

# Train the model

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=batch_size,nb_epoch = nb_epoch,show_accuracy=True,shuffle=True)
		  
 # Evaluate the model
score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1]) 



# Plot the results
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

# Confusion Matrix

Y_pred = model.predict(X_val_new)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
  

y_pred = model.predict_classes(X_val_new)
print(y_pred)

p=model.predict_proba(X_val_new) # to predict probability

target_names = ['class 0(non nodule)', 'class 1(nodule)']
print(classification_report(np.argmax(y_val_new,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_val_new,axis=1), y_pred))




# saving weights

fname = "myDataSet/patches3D/weights-Test-3DCNNKeras.hdf5"
model.save_weights(fname,overwrite=True)














 
