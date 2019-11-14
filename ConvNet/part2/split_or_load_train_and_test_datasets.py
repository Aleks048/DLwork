from os.path import exists
from numpy import load,asarray,reshape,save,empty,float64,shape,delete
from json import load as jload
import CONSTANTS as CONST
from random import shuffle

'''
tested all worked
'''


def extractDictFromJSON(path:str):
    '''
    opens jSON as dictionary
    '''
    with open(path,'r') as df:
        data = jload(df)
        return data 


def splitTrainTestData(dictPath,splitPercent):
    '''
    to split data into train and test 
    '''
    data = extractDictFromJSON(dictPath)

    train = {}
    test = {}
    #counter = 0 #to be deleted????
    numOfDataPoints = len(data)
    
    dataKeys = list(data.keys())
    shuffle(dataKeys)
    trainKeys = dataKeys[:int(numOfDataPoints*splitPercent)]
    testKeys = dataKeys[int(numOfDataPoints*splitPercent):]

    for k,v in data.items():   
        if k in trainKeys:
            train[k] = v
        elif k in testKeys:
            test[k] = v
        #counter+=1#to be deleted?

    return train,test

def addingNumCracksFromSpreadsheet():
    import json
    import ntpath #for path split
    from sxl import Workbook
    from create_dataset_of_predictions import fileNameToConsistentFormat
    '''
    may fail on Linux because of ntpath/ but who cares?
    '''
    out = {}

    with open(CONST.jsonDatabasePath,'r') as data:
        jfile = json.load(data)
   
    #collecting number of cracks from the datasheet
    book = Workbook(CONST.driSpreadsheetPath)
    sheets = book.sheets
    sheetNames = sheets.keys()
    for k,jv in jfile.items():
        print("hi")
        f = [fileNameToConsistentFormat(ntpath.split(k)[1][0:3]),fileNameToConsistentFormat(ntpath.split(k)[1][4:-4])]
        if f[0] not in sheetNames:

            raise("NO spreadsheet with this name: ",f[0])
        else:
            currSheet = sheets[f[0]]
            for i,v in enumerate(currSheet.rows):
                if fileNameToConsistentFormat(str(v[1])) == f[1]:
                    yellow = int(v[4]) if type(v[4])==float else 0
                    red = int(v[14]) if type(v[14])==float else 0
                    blue = int(v[6]) if type(v[6])==float else 0
                    green = int(v[8]) if type(v[8])==float else 0
                    
                    out[k] = [jv[0],[yellow,red,blue,green],jv[1]]
                    break
    #dumping into the file
    with open(CONST.jsonWithNumCracksPath,"w") as f:
        json.dump(out,f)

def creatingTrDataAndTrLabels():
    '''
    creating numpy arrays of train and test datasets and labels if they don't exist
 
    '''   
    #if Numpy arrays already created
    if exists(CONST.part2npTrainDatasetPath) and exists(CONST.part2npTrainLabelsPath) and exists(CONST.part2npTestDatasetPath) and exists(CONST.part2npTestLabelsPath):
        trData = load(CONST.part2npTrainDatasetPath)
        trLabels  = load(CONST.part2npTrainLabelsPath)
        testData = load(CONST.part2npTestDatasetPath)
        testLabels  = load(CONST.part2npTestLabelsPath)

    #if Numpy arrays not created
    else:
        train,test = splitTrainTestData(CONST.jsonDatasetPath,CONST.largeDatasetTrTestingSplit)
        
      

        shapeX = CONST.rescaleImageX//CONST.part2PredictionStrideX
        shapeY = CONST.rescaleImageY//CONST.part2PredictionStrideY

        trLabels = empty(len(train))
        trNumCracks = empty((len(train),4))
        trData = empty(shape=(CONST.numOfColours,len(train),shapeX,shapeY,1),dtype=float64)
        count = 0
        for k in train:
            for c in range(CONST.numOfColours):
                

                if CONST.useDataset1:
                    trNumpy = asarray(train[k][2][c])#if we count the number of cracks from the spreadsheet
                    trNumpy = delete(trNumpy,1,1)#dropping the second column since predictions from part 1 have 2 numbers
                else:
                    trNumpy = asarray(train[k][1][c])
                trNumpy = reshape(trNumpy,newshape=(shapeX,shapeY,1))
                
                trData[c][count]=trNumpy
            
          
            try: 
                trLabels[count] = float(train[k][0])
            except:
                print(train[k][0])
                print()
            if CONST.useDataset1:
                trNumCracks[count] = train[k][1]
            else:
                trNumCracks[count] = train[k][0]
            count+=1
      
        testLabels = empty(len(test))
        testNumCracks = empty((len(test),4))
        testData = empty(shape=(CONST.numOfColours,len(test),shapeX,shapeY,1),dtype=float64)
        count = 0
        for k in test:
            for c in range(CONST.numOfColours):
                if CONST.useDataset1:
                    testNumpy = asarray(test[k][2][c])
                    testNumpy = delete(trNumpy,1,1)#dropping the second column
                else:
                    testNumpy = asarray(test[k][1][c])
                testNumpy = reshape(trNumpy,newshape=(shapeX,shapeY,1))
                
                testData[c][count]=trNumpy
            
            try: 
                testLabels[count] = test[k][0]               
            except:
                print(test[k][0])
                print()
            
            if CONST.useDataset1:
                testNumCracks[count] = test[k][1]
            else:
                testNumCracks[count] = test[k][0]
            count+=1

        #saving the datasets and labels
        save(CONST.part2npTrainDatasetPath,trData)
        save(CONST.part2npTrainLabelsPath,trLabels)
        save(CONST.part2npTrainNumCracksPath,trNumCracks)
        save(CONST.part2npTestDatasetPath,testData)
        save(CONST.part2npTestLabelsPath,testLabels)
        save(CONST.part2npTestNumCracksPath,testNumCracks)
    return trData,trLabels,testData,testLabels


