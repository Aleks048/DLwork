from __future__ import print_function
import keras
from keras.backend.tensorflow_backend import set_session

import tensorflow

from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt

#from PIL import Image

#import xlrd

#import os
import numpy
import math

import tensorflow as tf

import gc

from random import shuffle

import gridImages
from DataGenerator import DataGenerator
import preprocessingPaintedCracks as pr


def flattenList(list):
    out = []
    for l in list:
        for e in l:
            out.append(e)
    return out

def alexNetModel(input_shape,num_classes):
    model = Sequential()

    model.add(Conv2D(96,kernel_size=11,strides=4,input_shape=input_shape,activation="elu",padding='valid'))
    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(256,kernel_size=5,strides=1,padding="same",activation="elu"))
    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"))
                

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu"))



    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2()))
    model.add(keras.layers.Dropout(0.5))
    model.add(Dense(4096, activation='elu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def alexExperimentNetModel(input_shape,num_classes,drop1,drop2,regularizerRate1,regularizerRate2,initializer):
    model = Sequential()

    model.add(Conv2D(96,kernel_size=11,strides=4,input_shape=input_shape,activation="elu",padding='valid',kernel_initializer=initializer,))
    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(256,kernel_size=5,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(keras.layers.BatchNormalization())

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
                

    model.add(Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))



    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate1),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop1))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(regularizerRate2),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop2))
    model.add(Dense(1000, activation='relu',kernel_initializer=initializer,bias_initializer="zeros"))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model

"""
def alexExperimentNetModelVGG19(input_shape,num_classes,drop1,drop2,initializer):
    model = Sequential()

    model.add(Conv2D(64,kernel_size=3,strides=1,input_shape=input_shape,activation="elu",padding='valid',kernel_initializer=initializer,))
    
    model.add(keras.layers.BatchNormalization())

    #model.add(Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(64,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(Conv2D(128,kernel_size=3,strides=1,padding="same",activation="relu",kernel_initializer=initializer))
    model.add(Conv2D(128,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))

    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))


    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))
    model.add(Conv2D(512,kernel_size=3,strides=1,padding="same",activation="elu",kernel_initializer=initializer))

    model.add(MaxPooling2D(pool_size=3,strides=2))
    


    model.add(Flatten())
    #model.add(keras.layers.Dropout(0.85))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop1))
    model.add(Dense(4096, activation='elu',kernel_regularizer=keras.regularizers.l2(),kernel_initializer=initializer,bias_initializer="zeros"))
    model.add(keras.layers.Dropout(drop2))
    model.add(Dense(1000, activation='relu',kernel_initializer=initializer,bias_initializer="zeros"))

    model.add(Dense(num_classes, activation='softmax'))

    return model
"""





#param param param
img_x,img_y = 180, 180
numOfChannels = 3
numOfImagesPerDir = 15
numOfKFolds = 5
epochs = 50
batch = 96

numClasses = 2

numNoCrack =0
numYellowCrack = 3
numRedCrack = 1
numBlueCrack  = 2
numGreenCrack = 4
#numXCrack = 0
#numYcrack = 1



alpha = 1.3/pow(5,8)#1.3/pow(5,6)#6
beta_1 = 0.9
beta_2 = 0.99
decay = (1/epochs)*alpha
epsilon = 10e-8
amsGrad = True
modelKernelInitializer = "he_uniform"
#optimizer = keras.optimizers.SGD(lr =alpha,momentum = 0.99,nesterov =True)
optimizer = keras.optimizers.Nadam(lr=alpha, beta_1=beta_1, beta_2=beta_2, epsilon=None, schedule_decay=decay)
#optimizer=keras.optimizers.Adam(lr=alpha,beta_1=beta_1,beta_2=beta_2,decay=decay,epsilon = epsilon,amsgrad=amsGrad)
loss = keras.losses.binary_crossentropy

drop1 =0.6
drop2 = 0.6
regRate1=0.2
regRate2=0.1


input_shape = (img_x,img_y,numOfChannels)

readyFolders = ["A","B","C","D"]
dataAugTypes = ["0rot","90rot","180rot","270rot","0ref","90ref","180ref","270ref"]



y_train=numpy.empty(0)
y_test=numpy.empty(0)

x_train=numpy.empty((0, img_x, img_y, numOfChannels))
x_test=numpy.empty((0, img_x, img_y, numOfChannels))



#create x and y lists
imagesEvalListCracks = []
imagesEvalListNoCracks = []

for augType in dataAugTypes:
    CrackImgList = gridImages.getTheImagesWithCracksList(readyFolders,numOfImagesPerDir,augType)
    NoCrackImgList = gridImages.createImagesWithoutCracksList(CrackImgList,False,readyFolders,augType)
    

    imagesEvalListCracks.append(CrackImgList)
    imagesEvalListNoCracks.append(NoCrackImgList)


imagesEvalListCracks = flattenList(imagesEvalListCracks)
imagesEvalListNoCracks = flattenList(imagesEvalListNoCracks)

shuffle(imagesEvalListCracks)
shuffle(imagesEvalListNoCracks)




imgList = pr.getMyIMagesList(pr.listOfColorsUsed,pr.folderNames,r"C:\Users\ytr16\source\repos\ConvNet\painted_data\\",dataAugTypes)

shuffle(imgList)


#print(len(imagesListNoCracks))
#print(len(imagesListCracks))
#print(len(imagesListNoCracks))



#numOfImagesPerFold = len(imagesListCracks)//numOfKFolds
numOfImagesPerFold = len(imgList)//numOfKFolds

#with tensorflow.device("/gpu:0"):
for k in range(numOfKFolds):
    partition = {'train':[],"test":[],"eval":[]}
    y_labels_train = {}
    y_labels_test = {}
    y_labels_eval = {}
    #testImagesList = imagesListCracks[k*numOfImagesPerFold:(k+1)*numOfImagesPerFold]
    #testImagesList += imagesListNoCracks[k*numOfImagesPerFold:(k+1)*numOfImagesPerFold]


    evalImagesList = imgList[k*numOfImagesPerFold:(k+1)*numOfImagesPerFold]
    trainImagesList = imgList[:k*numOfImagesPerFold]
    trainImagesList += imgList[(k+1)*numOfImagesPerFold:]

    testImagesList = imagesEvalListCracks+imagesEvalListNoCracks
    shuffle(testImagesList)

   
    #trainImagesList = imagesListCracks[:k*numOfImagesPerFold]
    #trainImagesList += imagesListCracks[(k+1)*numOfImagesPerFold:]
    #trainImagesList += imagesListNoCracks[:k*numOfImagesPerFold]
    #trainImagesList += imagesListNoCracks[(k+1)*numOfImagesPerFold:]

    #shuffle(trainImagesList)
    #shuffle(testImagesList)

     #making the classes equal
    countNoCracks = [i for i in imgList if i.hasTheCrack=="noCracks"]
    countBlueCracks = [i for i in imgList if i.hasTheCrack=="blue"]
    countRedCracks =  [i for i in imgList if i.hasTheCrack=="red"]
    countYellowCracks =  [i for i in imgList if i.hasTheCrack=="yellow"]
    shuffle(countNoCracks)
    print(len(countBlueCracks))
    print(len(countRedCracks))
    imgList=countNoCracks[:len(countRedCracks)*3]+countBlueCracks+countRedCracks
    #print(len(countBlueCracks))
    shuffle(imgList)
    trainImagesList = imgList
    

    #eval images
    for img in testImagesList:
        #print(img.hasTheCrack)
        if img.hasTheCrack=="No":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numNoCrack
        if img.hasTheCrack=="XX2":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numRedCrack
            #print(img.name)
        elif img.hasTheCrack=="YY1":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numNoCrack
            # print(img.name)

    #train test images
    for img in trainImagesList: 
        #print(img.hasTheCrack)
        if img.hasTheCrack=="noCracks":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numNoCrack
        #elif img.hasTheCrack=="yellow":
        #    partition["train"].append(img.dir+img.name)
        #    y_labels_train[img.dir+img.name] = numYellowCrack
        if img.hasTheCrack=="red":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numRedCrack
        elif img.hasTheCrack=="blue":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numNoCrack
        #elif img.hasTheCrack=="green":
        #    partition["train"].append(img.dir+img.name)
        #    y_labels_train[img.dir+img.name] = numGreenCrack
        

    #for img in evalImagesList:
    #    if img.hasTheCrack=="noCracks":
    #        partition["eval"].append(img.dir+img.name)
    #        y_labels_eval[img.dir+img.name] = numNoCrack
        #elif img.hasTheCrack=="yellow":
        #    partition["eval"].append(img.dir+img.name)
        #    y_labels_eval[img.dir+img.name] = numYellowCrack
        #if img.hasTheCrack=="red":
        #    partition["eval"].append(img.dir+img.name)
        #    y_labels_eval[img.dir+img.name] = numRedCrack
        #elif img.hasTheCrack=="blue":
        #    partition["eval"].append(img.dir+img.name)
        #    y_labels_eval[img.dir+img.name] = numBlueCrack
        #elif img.hasTheCrack=="green":
        #    partition["eval"].append(img.dir+img.name)
        #    y_labels_eval[img.dir+img.name] = numGreenCrack


        

    params = {'dim': (img_x,img_y),
                            'batch_size': batch,
                            'n_classes': numClasses,
                            'n_channels': numOfChannels,
                            'shuffle': True}

    training_generator = DataGenerator(partition['train'], y_labels_train, **params)         
    test_generator = DataGenerator(partition['test'], y_labels_test, **params)
    eval_generator = DataGenerator(partition['eval'],y_labels_eval,**params)


    model = alexExperimentNetModel(input_shape,numClasses,drop1,drop2,regRate1,regRate2,modelKernelInitializer)

    
    print(alpha)
    
                
    model.compile(loss=loss,
                    optimizer = optimizer
                    ,metrics=['accuracy'])

    model.fit_generator(generator=training_generator,
                                    validation_data=test_generator,
                                    #steps_per_epoch=len(partition["train"])//batch,
                                    epochs=epochs,
                                    use_multiprocessing=False,
                                    workers=32,
                                    max_queue_size=300,
                                    verbose=1
                                    )            
      
    scores = model.evaluate_generator(generator=test_generator,verbose=1)
    print("loss:"+str(scores[0])+" acc:"+str(scores[1]))

    model.save("C:\\Users\\ytr16\\source\\repos\\ConvNet\\model"+str(k+1)+".h5")                            
    del model
    
    gc.collect()

    #tf.keras.backend.clear_session()
    #keras.backend.clear_session()