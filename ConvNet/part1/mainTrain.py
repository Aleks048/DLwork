'''
here the training for part 1 is happening
'''

import keras
import tensorflow as tf#used in the cleaning after the model

import os#used in check if we continue training or start new one

from DataGenerator import DataGenerator#feeds the data into the model
from preprocessing.utils.plotAccAndLoss import plotAccAndLoss 

from keras.applications import VGG19#if the pretrained model is used

import CONSTANTS_part1 as CONST_p1#all ther constants required for training

def trainTheModelPart1(trData,trLabels,testData,testLabels,cv,model):
    '''
    the network or networks for part 1 are trained here 
    '''

    if CONST_p1.part1Debuging:#show log if debuging
        from utils.logGeneration import findHowManyImagesInEachClass
        findHowManyImagesInEachClass(trData,"train")
        findHowManyImagesInEachClass(testData,"test")

    optimizer = CONST_p1.optimizer
    loss = CONST_p1.loss

    trWeights = [CONST_p1.part1weightDict[trLabels[i]] for i in trLabels]#creating array of weights for the data since the data is highly unbalanced
   
    #create data generator
    if CONST_p1.part1usePretrainedNetwork:
        params = CONST_p1.data_generator_params_pretrainedNet
    else:
        params = CONST_p1.data_generator_params
    training_generator = DataGenerator(trData, trLabels, **params)
    test_generator = DataGenerator(testData, testLabels, **params)

    #crete model
    if os.path.exists(CONST_p1.part1trainedModelSavePath):#check if the model already exists
        model = keras.models.load_model(CONST_p1.part1trainedModelSavePath)
    else:
        if CONST_p1.part1usePretrainedNetwork:
            model = CONST_p1.model_pretrainedNet
        else:
            model  = CONST_p1.model

    #compile
    model.compile(loss=loss,
                optimizer = optimizer
                ,metrics=['accuracy'])

    #fit
    history = model.fit_generator(generator=training_generator,
                                    validation_data=test_generator,
                                    class_weight=trWeights,
                                    **CONST_p1.training_parameters
                                    #callbacks=[keras.callbacks.EarlyStopping(monitor= 'val_acc',patience=3,mode = "max",baseline=80)]#if early stopping used
                                    )    
    
    #save and plot
    model.save(CONST_p1.part1trainedModelSavePath)
    plotAccAndLoss(history,cv)

    #clean
    del(history)
    model.reset_states()
    del(model)
    tf.reset_default_graph()
    keras.backend.clear_session()


if __name__ == "__main__":

    import numpy
    import sys
    import CONSTANTS as CONST
    import json

    #!!!ALERT!!! run preprocessingPainedCracks before to get the names correctly
    if not CONST_p1.useCrossValidation:
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



