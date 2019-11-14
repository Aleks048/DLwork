'''
here the training for part 1 is happening
'''


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
import os
import time


from random import shuffle


import sys


import CONSTANTS as CONST
from MyImage import MyImage

from DataGenerator import DataGenerator
import part1networks
from keras.applications import VGG19


import json
'''
#def flattenList(list):
#    out = []
#    for l in list:
#        for e in l:
#            out.append(e)
#    return out
'''

def reset_keras():
    from keras.backend.tensorflow_backend import set_session
    from keras.backend.tensorflow_backend import clear_session
    from keras.backend.tensorflow_backend import get_session
    import tensorflow

    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    #try:
    #    del classifier # this is from global space - change this as you need
    #except:
    #    pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))


def train(numOfEpochs,pathToWeights,loadedData,dataSplittedByColor,trainColorNum):
    #param param param
    img_x,img_y = 30, 30
    numOfChannels = 3
    input_shape = (img_x,img_y,numOfChannels)
    numOfKFolds = 4
    epochs = numOfEpochs
    batch = 100


    drop1 =0.8
    drop2 = 0.8
    regRate1=0.1
    regRate2=0.1

    alpha = 1.3/pow(5,7)#6
    beta_1 = 0.9
    beta_2 = 0.99
    decay = (1/epochs)*alpha
    epsilon = 10e-8
    amsGrad = True
    modelKernelInitializer="he_uniform"

    optimizer=keras.optimizers.Adam(lr=alpha,beta_1=beta_1,beta_2=beta_2,decay=decay,epsilon = epsilon,amsgrad=amsGrad)
    loss = keras.losses.mse
    #



    theTrainingColorNum = trainColorNum

    numClasses = 2 if theTrainingColorNum != -1 else 5

    numNoCrack =1 if theTrainingColorNum==0 else (0 if theTrainingColorNum==-1 else 0)
    numYellowCrack = 1 if theTrainingColorNum==1 else (1 if theTrainingColorNum==-1 else 0)
    numRedCrack = 1 if theTrainingColorNum==2 else (2 if theTrainingColorNum==-1 else 0)
    numBlueCrack  = 1 if theTrainingColorNum==3 else (3 if theTrainingColorNum==-1 else 0)
    numGreenCrack = 1 if theTrainingColorNum==4 else (4 if theTrainingColorNum==-1 else 0)



    #check if all the data is included
    if len(pr.augTypes)!=8:#check if have all augmented datatypes
        raise Exception("The list of augmented datatypes is not full")
    if len(pr.folderNames)!=5:#check if have all folders    
        raise Exception("The list of folders is not full")
        pass
    if len(pr.listOfColorsUsed)!=4:#check if have all colors
        raise Exception("The list of colors used is not full")


    imgList = loadedData#pr.getMyIMagesList(pr.listOfColorsUsed,pr.folderNames,r"S:\convnet_smaller_images\ConvNet\painted_data\\",pr.augTypes)


    numOfImagesPerFold = len(imgList)//numOfKFolds


    #separating the images by color
    countNoCracks = dataSplittedByColor[0]
    countYellowCracks = dataSplittedByColor[1]
    countRedCracks =  dataSplittedByColor[2]
    countBlueCracks = dataSplittedByColor[3]
    countGreenCracks =  dataSplittedByColor[4]


    lenCountNoCracks = len(countNoCracks)
    lenCountYellowCracks = len(countYellowCracks)
    lenCountRedCracks = len(countRedCracks)
    lenCountBlueCracks = len(countBlueCracks)
    lenCountGreenCracks = len(countGreenCracks)




    trTestDivisorNoCracks =  lenCountNoCracks-lenCountNoCracks//numOfKFolds
    trTestDivisorYellowCracks =  lenCountYellowCracks-lenCountYellowCracks//numOfKFolds
    trTestDivisorRedCracks =  lenCountRedCracks-lenCountRedCracks//numOfKFolds
    trTestDivisorBlueCracks =  lenCountBlueCracks-lenCountBlueCracks//numOfKFolds
    trTestDivisorGreenCracks =  lenCountGreenCracks-lenCountGreenCracks//numOfKFolds

    print()
    if not(os.path.exists(pathToWeights)):
        shuffle(countNoCracks)
        shuffle(countRedCracks)
        shuffle(countBlueCracks)
        shuffle(countYellowCracks)
        shuffle(countGreenCracks)

    

    #trainImagesList = countNoCracks[:len(countNoCracks)-(k+1)*len(countNoCracks)//numOfKFolds]+countYellowCracks[:len(countYellowCracks)-(k+1)*len(countYellowCracks)//numOfKFolds]+countBlueCracks[:len(countBlueCracks)-(k+1)*len(countBlueCracks)//numOfKFolds]+countRedCracks[:len(countRedCracks)-(k+1)*len(countRedCracks)//numOfKFolds]+countGreenCracks[:len(countGreenCracks)-(k+1)*len(countGreenCracks)//numOfKFolds]
    #testImagesList = countNoCracks[len(countNoCracks)-(k+1)*len(countNoCracks)//numOfKFolds:]+countYellowCracks[len(countYellowCracks)-(k+1)*len(countYellowCracks)//numOfKFolds:]+countBlueCracks[len(countBlueCracks)-(k+1)*len(countBlueCracks)//numOfKFolds:]+countRedCracks[len(countRedCracks)-(k+1)*len(countRedCracks)//numOfKFolds:]+countGreenCracks[len(countGreenCracks)-(k+1)*len(countGreenCracks)//numOfKFolds:]
    
    lengthOfTheDataSet = 0 
  
    trainImagesList = [] 
    testImagesList = []

    numOfSecondClassDivisions = 4
    trainMin = min([i for i in [trTestDivisorYellowCracks,trTestDivisorRedCracks,trTestDivisorBlueCracks,trTestDivisorGreenCracks,trTestDivisorNoCracks] if i != 0])
    testMin = min([i for i in [len(countRedCracks),len(countBlueCracks),len(countGreenCracks),len(countNoCracks),len(countYellowCracks)] if i != 0])-trainMin

    trainMin //= numOfSecondClassDivisions
    testMin //= numOfSecondClassDivisions 


    print([trTestDivisorRedCracks,trTestDivisorBlueCracks,trTestDivisorGreenCracks,trTestDivisorNoCracks])
    print(trainMin)
    print(testMin)

   

    if theTrainingColorNum==-1:
        pass
       # trainImagesList = countNoCracks[:len(countNoCracks)-(k+1)*len(countNoCracks)//numOfKFolds]+countYellowCracks[:len(countYellowCracks)-(k+1)*len(countYellowCracks)//numOfKFolds]+countBlueCracks[:len(countBlueCracks)-(k+1)*len(countBlueCracks)//numOfKFolds]+countRedCracks[:len(countRedCracks)-(k+1)*len(countRedCracks)//numOfKFolds]+countGreenCracks[:len(countGreenCracks)-(k+1)*len(countGreenCracks)//numOfKFolds]
       # testImagesList = countNoCracks[len(countNoCracks)-(k+1)*len(countNoCracks)//numOfKFolds:]+countYellowCracks[len(countYellowCracks)-(k+1)*len(countYellowCracks)//numOfKFolds:]+countBlueCracks[len(countBlueCracks)-(k+1)*len(countBlueCracks)//numOfKFolds:]+countRedCracks[len(countRedCracks)-(k+1)*len(countRedCracks)//numOfKFolds:]+countGreenCracks[len(countGreenCracks)-(k+1)*len(countGreenCracks)//numOfKFolds:]
    if theTrainingColorNum==0:#NO 
        #alpha *= 5#5??
        ##optimizer=keras.optimizers.Adam(lr=alpha,beta_1=beta_1,beta_2=beta_2,decay = 0.0,epsilon = epsilon,amsgrad=amsGrad)#use optimizer without decay
        #optimizer = keras.optimizers.Adamax(lr=alpha, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        ##optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

        ##trainImagesList = countYellowCracks[:trTestDivisorYellowCracks]+countRedCracks[:trTestDivisorRedCracks]+countBlueCracks[:trTestDivisorBlueCracks]+countNoCracks[:trTestDivisorNoCracks]+countGreenCracks[:trTestDivisorGreenCracks]
        ##testImagesList = countYellowCracks[trTestDivisorYellowCracks:]+countRedCracks[trTestDivisorRedCracks:]+countBlueCracks[trTestDivisorBlueCracks:]+countNoCracks[trTestDivisorNoCracks:]+countGreenCracks[trTestDivisorGreenCracks:]
   
   
        #trainImagesList = countNoCracks[:trTestDivisorYellowCracks]+countYellowCracks[:trTestDivisorYellowCracks]
        #testImagesList = countNoCracks[trTestDivisorYellowCracks:len(countYellowCracks)]+countYellowCracks[trTestDivisorYellowCracks:]
        pass

    if theTrainingColorNum==1:#Yellow 
        numOfSecondClassDivisions = 4


        trainImagesList = countYellowCracks[:trainMin*numOfSecondClassDivisions]+countRedCracks[:trainMin]+countBlueCracks[:trainMin]+countGreenCracks[:trainMin]+countNoCracks[:trainMin]
        testImagesList = countYellowCracks[trainMin*numOfSecondClassDivisions:trainMin*numOfSecondClassDivisions+testMin*numOfSecondClassDivisions]+countRedCracks[trainMin:trainMin+testMin]+countBlueCracks[trainMin:trainMin+testMin]+countGreenCracks[trainMin:trainMin+testMin]+countNoCracks[trainMin:trainMin+testMin]
   
    if theTrainingColorNum==2:#Red
        numOfSecondClassDivisions = 4


        trainImagesList = countRedCracks[:trainMin*numOfSecondClassDivisions]+countYellowCracks[:trainMin]+countBlueCracks[:trainMin]+countGreenCracks[:trainMin]+countNoCracks[:trainMin]
        testImagesList = countRedCracks[trainMin*numOfSecondClassDivisions:trainMin*numOfSecondClassDivisions+testMin*numOfSecondClassDivisions]+countYellowCracks[trainMin:trainMin+testMin]+countBlueCracks[trainMin:trainMin+testMin]+countGreenCracks[trainMin:trainMin+testMin]+countNoCracks[trainMin:trainMin+testMin]
   

    if theTrainingColorNum==3:#Blue
        numOfSecondClassDivisions = 4

        trainImagesList = countBlueCracks[:trainMin*numOfSecondClassDivisions]+countRedCracks[:trainMin]+countYellowCracks[:trainMin]+countGreenCracks[:trainMin]+countNoCracks[:trainMin]
        testImagesList = countBlueCracks[trainMin*numOfSecondClassDivisions:trainMin*numOfSecondClassDivisions+testMin*numOfSecondClassDivisions]+countYellowCracks[trainMin:trainMin+testMin]+countRedCracks[trainMin:trainMin+testMin]+countGreenCracks[trainMin:trainMin+testMin]+countNoCracks[trainMin:trainMin+testMin]
    

    if theTrainingColorNum==4:#Green
        numOfSecondClassDivisions = 4

        trainImagesList = countGreenCracks[:trainMin*numOfSecondClassDivisions]+countRedCracks[:trainMin]+countYellowCracks[:trainMin]+countBlueCracks[:trainMin]+countNoCracks[:trainMin]
        testImagesList = countGreenCracks[trainMin*numOfSecondClassDivisions:trainMin*numOfSecondClassDivisions+testMin*numOfSecondClassDivisions]+countYellowCracks[trainMin:trainMin+testMin]+countRedCracks[trainMin:trainMin+testMin]+countBlueCracks[trainMin:trainMin+testMin]+countNoCracks[trainMin:trainMin+testMin]
    

    #dataset consistency testing
    for im in trainImagesList:
        for tIm in testImagesList:
            if im.name==tIm.name:
                raise Exception ("The train and test datasets are blended!")

    if trainImagesList==[] or testImagesList == []:
        raise Exception("the length of your dataset is 0!")


    f= open("S:\convnet_smaller_images\ConvNet\painted_data\\"+"accuracy.txt","a")
    
    f.write(sys.argv[1]+"\n")

    f.write("original test images number:"+str(len(testImagesList))+"\n")
    f.write("original train images number:"+str(len(trainImagesList))+"\n")


    f.write("#test yellow color images : "+str(len([i for i in testImagesList if i.hasTheCrack=="yellow"]))+"\n")
    f.write("#train yellow color images: "+str(len([i for i in trainImagesList if i.hasTheCrack=="yellow"]))+"\n")


    f.write("#test red color images : "+str(len([i for i in testImagesList if i.hasTheCrack=="red"]))+"\n")
    f.write("#train red color images: "+str(len([i for i in trainImagesList if i.hasTheCrack=="red"]))+"\n")


    f.write("#test blue color images : "+str(len([i for i in testImagesList if i.hasTheCrack=="blue"]))+"\n")
    f.write("#train blue color images: "+str(len([i for i in trainImagesList if i.hasTheCrack=="blue"]))+"\n")


    f.write("#test green color images : "+str(len([i for i in testImagesList if i.hasTheCrack=="green"]))+"\n")
    f.write("#train green color images: "+str(len([i for i in trainImagesList if i.hasTheCrack=="green"]))+"\n")


    f.write("#test prime NOcolor images : "+str(len([i for i in testImagesList if i.hasTheCrack=="noCracks"]))+"\n")
    f.write("#train prime NOcolor images: "+str(len([i for i in trainImagesList if i.hasTheCrack=="noCracks"]))+"\n")

    f.close()
    #adding the augTypes images
    for im in trainImagesList:
        if im in imgList:
            for augIm in imgList[im]:
                trainImagesList.append(augIm)
    for im in testImagesList:
        if im in imgList:
            for augIm in imgList[im]:
                testImagesList.append(augIm)

    if not(os.path.exists(pathToWeights)):
        shuffle(trainImagesList)
        shuffle(testImagesList)


    #collecting data into the datagenerators
    partition = {'train':[],"test":[],"eval":[]}
    y_labels_train = {}
    y_labels_test = {}
    #test images
    for img in testImagesList:
        #print(img.hasTheCrack)
        if img.hasTheCrack=="noCracks":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numNoCrack
        if img.hasTheCrack=="red":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numRedCrack
            #print(img.name)
        if img.hasTheCrack=="yellow":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numYellowCrack
            #print(img.name)
        if img.hasTheCrack=="blue":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numBlueCrack
            #print(img.name)
        if img.hasTheCrack=="green":
            partition["test"].append(img.dir+img.name)
            y_labels_test[img.dir+img.name] = numGreenCrack
            #print(img.name)

    #train test images
    for img in trainImagesList: 
        #print(img.hasTheCrack)
        if img.hasTheCrack=="noCracks":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numNoCrack
        if img.hasTheCrack=="red":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numRedCrack
        if img.hasTheCrack=="yellow":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numYellowCrack
        if img.hasTheCrack=="blue":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numBlueCrack
        if img.hasTheCrack=="green":
            partition["train"].append(img.dir+img.name)
            y_labels_train[img.dir+img.name] = numGreenCrack
        
        

    params = {'dim': (img_x,img_y),
                            'batch_size': batch,
                            'n_classes': numClasses,
                            'n_channels': numOfChannels,
                            'shuffle': True}

    training_generator = DataGenerator(partition['train'], y_labels_train, **params)         
    test_generator = DataGenerator(partition['test'], y_labels_test, **params)



    model  = leNet(input_shape=input_shape,num_classes=numClasses)
    #if os.path.exists(pathToWeights):
    #    model = keras.models.load_model(pathToWeights)
    #else:
    #    model = alexExperimentNetModel(input_shape,numClasses,drop1,drop2,regRate1,regRate2,modelKernelInitializer)

    model.summary()
    
    print(alpha)
                 
    model.compile(loss=loss,
                    optimizer = optimizer
                    ,metrics=['accuracy'])

    history = model.fit_generator(generator=training_generator,
                                    validation_data=test_generator,
                                    #steps_per_epoch=len(partition["train"])//batch,
                                    epochs=epochs,
                                    use_multiprocessing=False,
                                    workers=32,
                                    max_queue_size=300,
                                    verbose=1,
                                    #callbacks=[keras.callbacks.EarlyStopping(monitor= 'val_acc',patience=3,mode = "max",baseline=80)]
                                    )    
    

      
    scores = model.evaluate_generator(generator=test_generator,verbose=1)
    print("loss:"+str(scores[0])+" acc:"+str(scores[1]))

    
    model.save(pathToWeights)         
    
    float_formatter = lambda x: "%.2f" % x

    f= open("S:\convnet_smaller_images\ConvNet\painted_data\\"+"accuracy.txt","a")
    f.write("acc: "+str(scores[1])+"\n")
    f.write("Training acc:\n")
    f.write("".join(str(float_formatter(i)+" ") for i in history.history["acc"]))
    f.write("\n Val acc:\n")
    f.write("".join(str(float_formatter(i)+" ") for i in history.history["val_acc"]))
    f.write("\n\n\n")
    f.close()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("S:\convnet_smaller_images\ConvNet\painted_data\\"+sys.argv[1]+".jpg")

    del history
    reset_keras()    
    

def trainTheModelPart1(trData,trLabels,testData,testLabels,cv,model):
    '''
    the network or networks for part 1 are trained here 
    '''
    import sys
    sys.path.insert(0, './part2')
    from part2networks import plotAccAndLoss

    from test_part1 import findHowManyImagesInEachClass

    if CONST.part1Debuging:
        findHowManyImagesInEachClass(trData)
        findHowManyImagesInEachClass(testData)

    alpha = 1.3/pow(5,7)#6
    beta_1 = 0.9
    beta_2 = 0.99
    decay = (1/100)*alpha
    epsilon = 10e-8
    amsGrad = True
    modelKernelInitializer="he_uniform"

    drop1 =0.8
    drop2 = 0.8
    regRate1=0.1
    regRate2=0.1


    #creating array of weights
    trWeights = [CONST.part1weightDict[trLabels[i]] for i in trLabels]
    #testWeights = [CONST.part1weightDict[testLabels[i]] for i in testLabels]

    optimizer=keras.optimizers.Adam(lr=alpha,beta_1=beta_1,beta_2=beta_2,decay=decay,epsilon = epsilon,amsgrad=amsGrad)
    loss = keras.losses.categorical_crossentropy

    if CONST.part1usePretrainedNetwork:
        params = {'dim': (512,0),
                 'batch_size': 30,
                 'n_classes':CONST.numClasses,
                 'n_channels': 0,
                 'shuffle': True,
                 'useConv': False
                 }
        inShape = (512)
    else:
         params = {'dim': (CONST.part1StrideX,CONST.part1StrideY),
                 'batch_size': 100,
                 'n_classes':CONST.numClasses,
                 'n_channels': CONST.part1NumChannels,
                 'shuffle': True,
                 'useConv': True
                 }
         inShape = (CONST.part1StrideX,CONST.part1StrideY,CONST.part1NumChannels)
    training_generator = DataGenerator(trData, trLabels, **params)
    test_generator = DataGenerator(testData, testLabels, **params)

    if os.path.exists(CONST.part1trainedModelSavePath):
        model = keras.models.load_model(CONST.part1trainedModelSavePath)
    else:
        model  = part1networks.leNetExp(input_shape=inShape,num_classes=CONST.numClasses)
    #model = part1networks.alexExperimentNetModelDeeper((30,30,3),CONST.numClasses,drop1,drop2,regRate1,regRate2,modelKernelInitializer)

    model.compile(loss=loss,
                    optimizer = optimizer
                    ,metrics=['accuracy'])

    history = model.fit_generator(generator=training_generator,
                                    validation_data=test_generator,
                                    
                                    epochs=3000,
                                    use_multiprocessing=False,
                                    workers=1,
                                    max_queue_size=1,
                                    verbose=1,
                                    class_weight=trWeights
                                    #callbacks=[keras.callbacks.EarlyStopping(monitor= 'val_acc',patience=3,mode = "max",baseline=80)]
                                    )    
    
    model.save(CONST.part1trainedModelSavePath)
    
    plotAccAndLoss(history,cv)
    del(history)
    model.reset_states()
    del(model)
    tf.reset_default_graph()
    keras.backend.clear_session()
    #reset_keras()


if __name__ == "__main__":

    if not CONST.useCrossValidation:
        if not(os.path.exists(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"trDataNames.npy") and 
               os.path.exists(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"trLabels.json") and
               os.path.exists(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"testDataNames.npy") and
               os.path.exists(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"testLabels.json")):
            from preprocessingPaintedCracks import saveDataNamesAndSplitTrTestFunctor
            saveDataNamesAndSplitTrTestFunctor().saveDataNamesAndSplitTrTest(CONST.listOfColorsUsed,CONST.folderNames,CONST.augTypes,CONST.part1ArraysOFDataNamesAndLabelsPath,0)
    
    
        trData = numpy.load(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"trDataNames.npy") 
        with open(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"trLabels.json") as f:
            trLabels = json.load(f)
        testData = numpy.load(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"testDataNames.npy") 
        with open(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"testLabels.json") as f:
            testLabels = json.load(f)
        trainTheModelPart1(trData,trLabels,testData,testLabels,0,"model")

    else:
        from preprocessingPaintedCracks import saveDataNamesAndSplitTrTestFunctor

       

        for cv  in range(CONST.numKFolds):
            saveData = saveDataNamesAndSplitTrTestFunctor()
            model = part1networks.leNetExp((30,30,3),CONST.numClasses)

            saveData.saveDataNamesAndSplitTrTest(CONST.listOfColorsUsed,CONST.folderNames,CONST.augTypes,CONST.part1ArraysOFDataNamesAndLabelsPath,float(cv))

            trData = numpy.load(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"trDataNames.npy") 
            with open(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"trLabels.json") as f:
                trLabels = json.load(f)
            testData = numpy.load(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"testDataNames.npy") 
            with open(CONST.part1ArraysOFDataNamesAndLabelsPath+"\\"+"testLabels.json") as f:
                testLabels = json.load(f)
            trainTheModelPart1(trData,trLabels,testData,testLabels,float(cv),"model")
#train(5000,r"S:\convnet_smaller_images\ConvNet\painted_data\\"+str(sys.argv[1])+".h5",imgList,[countNoCracks,countYellowCracks,countRedCracks,countBlueCracks,countGreenCracks],int(sys.argv[2])+1)



