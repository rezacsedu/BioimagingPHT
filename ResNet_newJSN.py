from __future__ import print_function

import numpy as np
from random import shuffle
import time
import csv
from PIL import Image
import os

import tensorflow as tf
import VGG
import keras
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import initializers
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.callbacks import CSVLogger
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
import sklearn.metrics as sklm

np.random.seed(3768)  # for reproducibility

# use this environment flag to change which GPU to use 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # make it 1 for CPU-based training 

#Get TensorFlow session
def get_session(): 
  config = tf.ConfigProto() 
  config.gpu_options.allow_growth = True 
  return tf.Session(config=config) 
  
# One hot encoding of labels 
def dense_to_one_hot(labels_dense,num_clases=5):
  return np.eye(num_clases)[labels_dense]

# Loading training MRI images
MRI_LabelPath = '/home/reza/DeepKneeExplainer/resources/newlabel.csv'
MRI_image_Path = '/home/reza/DeepKneeExplainer/resources/Data/MRI/unbalanced/up/'

def loadTrainImageAndLabels():
  imgList = []
  labelList = []
  readLabels = open(MRI_LabelPath) #label file path
  data = readLabels.readlines()
  
  files = os.listdir(MRI_image_Path + '/training/') #training path
  shuffle(files)
  
  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join(MRI_image_Path, file) #training path
        img = Image.open(fi_d).convert('L')

        im = np.array(img.resize((360,360), Image.ANTIALIAS))
        patient = file.split('_')[0]
        direction = file.split('_')[1].split('.')[0]
        label = "q"

        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label=row.split(",")[3]
              else:
                 label=row.split(",")[6]
              break
        if "V" in file:
                       label="3"
        if "8" not in label and "9" not in label and "X" not in label and '.' not in label:
          #if "." in label:
            #label='4'
          label= dense_to_one_hot(int(label),4)
          imgList.append(im)
          labelList.append(label)
  return np.array(imgList),np.array(labelList)

# Loading validation MRI images
def loadTestImages():
  imgList=[]
  labelList=[]
  readLabels = open(MRI_LabelPath) #label file path
  data = readLabels.readlines()
  files = os.listdir(MRI_image_Path + '/validation/') #test path

  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join(MRI_image_Path + '/validation/', file) #test path
        
        img = Image.open(fi_d).convert('L')
        im = np.array(img.resize((360,360), Image.ANTIALIAS))
        patient = file.split('_')[0]
        direction = file.split('_')[1].split('.')[0]
        label = "q"
        
        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label=row.split(",")[3]
              else:
                 label=row.split(",")[6]
              break
        if "V" in file:
                       label="3"
        if "8" not in label and "9" not in label and "X" not in label and '.' not in label:
          #if "." in label:
            #label='4'
          label= dense_to_one_hot(int(label),4)
          imgList.append(im)
          labelList.append(label)
  return np.array(imgList),np.array(labelList)

# Loading the labels of validation MRI images
def loadTestLabels(): #load labels with decimal format for comparison
  imgList = []
  labelList = []
  readLabels = open(MRI_LabelPath) #label path
  data = readLabels.readlines()
  files = os.listdir(MRI_image_Path + '/validation/') #test path
  
  for file in files:
        if file.endswith(".xml"):continue
        patient = file.split('_')[0]
        direction = file.split('_')[1].split('.')[0]
        label = "q"
        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label = row.split(",")[3]
              else:
                 label = row.split(",")[6]
              break
        if "V" in file:
                       label = "3"
        if "8" not in label and "9" not in label and "X" not in label and '.' not in label:
          #if "." in label:
            #label='4'
          labelList.append(int(label))
  return np.array(labelList)

#Defining hyperparameters
batch_size = 32

#Instantating VGG19 model
model = ResNet.ResNet18((360,360,1),4) #here you can choose ResNet18 34 50 101. The detail settings are input shape and class number

#Creating an optimizers
adaDelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95)
model.compile(optimizer = sgd , loss = 'mse',metrics = ['accuracy'])

#Image data generation for the training 
datagen = ImageDataGenerator(
        featurewise_center = False, 
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = True,  
        samplewise_std_normalization = False)  

# Preparing training and test sets
X_train, Y_train = loadTrainImageAndLabels()
X_test, Y_test = loadTestImages()

X_train = X_train.reshape( len(X_train), len(X_train[0]), len(X_train[0][0]),1)
X_test = X_test.reshape( len(X_test), len(X_test[0]), len(X_test[0][0]),1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

datagen.fit(X_train) 
for i in range(len(X_test)):
      X_test[i] = datagen.standardize(X_test[i])

#Creating early stopping 
earlystop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 50, verbose = 1, mode = 'auto', restore_best_weights = True)       

# Model training 
numEpochs = 100
steps_Per_Epoch
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), steps_per_epoch = steps_Per_Epoch, epochs = numEpochs, shuffle = True, validation_split = 0.20, verbose = 1)

from lossprettifier import LossPrettifier
reporter = LossPrettifier(show_percentage=True)

# visualizing losses and accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']
#print(train_loss)
#print(val_loss)

#Observing the losses but can be commented out as it's not mandatory 
for i in range(numEpochs-1):
    reporter(epoch=i, LossA = train_loss[i], LossAB = val_loss[i])

# Model evaluation 
score, acc = model.evaluate(X_test,Y_test,batch_size=batch_size)
print("Accuracy:",acc)

#if acc>0.675:
model.save_weights("VGG19-KLnew.h5")

y_pred = model.predict(X_test)
y_pred = y_pred.reshape( len(Y_test), 4)
Y_predict = y_pred.argmax(axis=-1)

# Writing results on file
f = open('ResNetRESULTS-JSNnewbalance-MRIup.txt','a') #create classification report
f.write(classification_report(loadTestLabels(), Y_predict))
f.write(str(sklm.cohen_kappa_score(loadTestLabels(), Y_predict))+","+str(acc)+","+str(score)+"\n")

#Print class-wise classification metrics
print(classification_report(loadTestLabels(), Y_predict))
