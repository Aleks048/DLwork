import sys
sys.path.insert(0, './part2')

from part2networks import *
from split_or_load_train_and_test_datasets import creatingTrDataAndTrLabels
import CONSTANTS as CONST


def testing(lr,createThePart1PredictedDataset:bool,part2NN):
    '''
    part 2

    lr : learning rate for the traning the model for part 2

    createThePart1PredictedDataset : whether the dataset using predictions from part 1 should be created or not
    '''
    if createThePart1PredictedDataset:
        from create_dataset_of_predictions import creatingPredictedDataset
        models = creatingPredictedDataset()

    trData,trLabels,testData,testLabels = creatingTrDataAndTrLabels()

    part2NN(trData,trLabels,testData,testLabels,True,lr)
      




   


NNsPart2 = [rawData,summedSimple58valAcc,findingModelWithAutokeras]
NNnumber = 1

if __name__ =="__main__":
        import tensorflow as tf
    
    #with tf.device("/gpu:"):
        print(len(sys.argv))
        if len(sys.argv)<=1 :
            testing(1.04e-5,CONST.createThePart1PredictedDataset,NNsPart2[NNnumber])
        else:
            testing(float(sys.argv[1]),CONST.createThePart1PredictedDataset,NNsPart2[NNnumber])